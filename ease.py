import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import math
import random
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import numpy as np
import copy
import time

# -------------------------
# Utilities
# -------------------------
def set_seed(seed=1993):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(device: str = "auto") -> torch.device:
    """
    Pick a compute device.
    - auto: prefer MPS (Apple Silicon), then CUDA, else CPU
    - mps/cuda/cpu: use if available, else fall back to CPU
    """
    if isinstance(device, torch.device):
        return device
    device = str(device).lower()
    if device in ("auto", "mps"):
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        if device == "mps":
            return torch.device("cpu")
        # auto fallback
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def cosine_sim(a, b, eps=1e-8):
    # a: (..., d), b: (..., d)
    an = a / (a.norm(dim=-1, keepdim=True) + eps)
    bn = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (an * bn).sum(dim=-1)

# -------------------------
# Backbone wrapper
# -------------------------
class FrozenBackbone(nn.Module):
    """
    Wrap a pretrained backbone and return a final embedding vector per input.
    Default: ResNet18 global pooling output (dim=512). You can swap to ViT.
    """
    def __init__(self, name='resnet18', pretrained=True, device='cpu'):
        super().__init__()
        if name == 'resnet18':
            # torchvision>=0.13 deprecates `pretrained=` in favor of `weights=`
            weights = None
            if pretrained:
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT
            b = torchvision.models.resnet18(weights=weights)
            # remove final fc
            modules = list(b.children())[:-1]  # up to avgpool
            self.fe = nn.Sequential(*modules)
            self.dim = b.fc.in_features
        else:
            raise NotImplementedError("Add other backbones if desired.")
        self.device = device
        # freeze parameters
        for p in self.fe.parameters():
            p.requires_grad = False
        self.to(device)

    def forward(self, x):
        # x -> batch x dim
        x = self.fe(x)  # B x C x 1 x 1
        x = x.view(x.size(0), -1)
        return x

# -------------------------
# Adapter module
# -------------------------
class Adapter(nn.Module):
    """
    Bottleneck adapter: down-project d->r, nonlin, up-project r->d, added as residual.
    This follows Eq. (4) of the paper in spirit.
    """
    def __init__(self, d, r=16):
        super().__init__()
        self.down = nn.Linear(d, r)
        self.act = nn.GELU()
        self.up = nn.Linear(r, d)

    def forward(self, x):
        # x: B x d
        return self.up(self.act(self.down(x)))

# -------------------------
# EASE manager
# -------------------------
class EASE_CIL:
    def __init__(self, backbone, device="auto", adapter_r=16, alpha=0.1):
        self.device = get_device(device)
        self.backbone = backbone
        self.adapters = []            # list of Adapter modules (one per task)
        self.adapter_r = adapter_r
        self.alpha = alpha
        self.class_to_task = {}       # map class label -> task index (0-based)
        self.prototypes = dict()      # prototypes[(task_index, class_label)] = tensor(d)
        self.seen_classes = []        # list of classes seen so far

    def new_task(self, task_classes):
        """
        Initialize a new adapter for an incoming task (task_classes is a list of class labels).
        """
        d = self.backbone.dim
        adapter = Adapter(d, r=self.adapter_r).to(self.device)
        self.adapters.append(adapter)
        t_idx = len(self.adapters) - 1
        for c in task_classes:
            self.class_to_task[c] = t_idx
            if c not in self.seen_classes:
                self.seen_classes.append(c)
        return t_idx

    def extract_features_all_adapters(self, loader):
        """
        For a given data loader, compute and return a dict:
          features[adapter_idx] = list of tensors features per sample (in order)
          labels = list of labels
        Implementation returns stacked tensors.
        """
        self.backbone.eval()
        for a in self.adapters:
            a.eval()
        features = {i: [] for i in range(len(self.adapters))}
        labels_list = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                base = self.backbone(x)  # B x d
                for i, adapter in enumerate(self.adapters):
                    # adapted embedding = base + adapter(base)
                    adapt = adapter(base)
                    emb = base + adapt
                    features[i].append(emb.cpu())
                labels_list.append(y.cpu())
        # stack
        for i in features:
            features[i] = torch.cat(features[i], dim=0)  # N x d
        labels = torch.cat(labels_list, dim=0)
        return features, labels

    def compute_prototypes_for_task(self, features, labels, task_classes):
        """
        features: dict adapter_idx -> N x d; labels: N
        Compute per-class prototype for classes in task_classes in each adapter subspace.
        Save into self.prototypes[(adapter_idx, class_label)].
        """
        for i in features:
            feats = features[i]
            for c in task_classes:
                mask = (labels == c)
                if mask.sum() == 0:
                    continue
                proto = feats[mask].mean(dim=0)
                self.prototypes[(i, c)] = proto

    def synthesize_old_prototypes(self, new_task_classes, new_adapter_idx):
        """
        Implements Eq. 8-9. For each old class o and new subspace (new_adapter_idx),
        we need:
          Po,o (old prototypes of old classes in old subspace) -- we use each old class's own adapter (its task adapter)
          Pn,o (new class prototypes in old subspace) -- compute by passing current new task data through old adapters (we do that during extraction)
          Pn,n (new class prototypes in new subspace)
        Then compute similarity Simi,j between Po,o[i] and Pn,o[j] in the co-occurrence (old) subspace,
        softmax over j, and produce P̂_o,n[i] = sum_j Simi,j * Pn,n[j].
        """
        # Build list of old classes
        old_classes = [c for c in self.seen_classes if self.class_to_task[c] < new_adapter_idx]
        if len(old_classes) == 0:
            return
        # For each old adapter (task), gather Po,o for old classes that belong to that adapter's task.
        # We'll synthesize per old class with respect to the *current* new subspace.
        # For simplicity we use the co-occurrence space as the *old class's adapter* (the adapter that produced Po,o).
        for old_c in old_classes:
            old_task = self.class_to_task[old_c]
            # Po,o is prototype of old_c in adapter old_task
            key_po = (old_task, old_c)
            if key_po not in self.prototypes:
                # can't synthesize if we don't have Po,o
                continue
            Po_o = self.prototypes[key_po]  # d
            # Build Pn,o: prototypes of each new class j in *old_task* adapter
            Pn_o_list = []
            new_cls_list = []
            for new_c in new_task_classes:
                k = (old_task, new_c)
                if k in self.prototypes:
                    Pn_o_list.append(self.prototypes[k])  # prototype of new class in old adapter space
                    new_cls_list.append(new_c)
            if len(Pn_o_list) == 0:
                continue
            Pn_o = torch.stack(Pn_o_list, dim=0)  # J x d
            # Pn,n: prototypes of new classes in the new adapter space
            Pn_n_list = []
            for new_c in new_cls_list:
                k2 = (new_adapter_idx, new_c)
                if k2 not in self.prototypes:
                    Pn_n_list.append(torch.zeros_like(Po_o))
                else:
                    Pn_n_list.append(self.prototypes[k2])
            Pn_n = torch.stack(Pn_n_list, dim=0)  # J x d

            # similarity Simi,j = cosine(Po,o[i], Pn,o[j]) ; softmax over j
            # Keep this on CPU: prototypes are stored on CPU, and mixing CPU/GPU here can break.
            Po_o_norm = Po_o.unsqueeze(0).to(Pn_o.dtype)  # 1 x d
            sims = F.cosine_similarity(Po_o_norm, Pn_o, dim=-1)  # J
            sims = F.softmax(sims, dim=0).to(Pn_n.dtype).unsqueeze(-1)  # J x 1

            # P_hat_o_n = sum_j sims_j * Pn_n[j]
            P_hat = (sims * Pn_n).sum(dim=0)
            # set synthesized prototype into self.prototypes for (new_adapter_idx, old_c)
            self.prototypes[(new_adapter_idx, old_c)] = P_hat

    def build_full_class_prototypes(self):
        """
        After adapters are added and prototypes (real or synthesized) exist, build
        a structure mapping class -> list of per-adapter prototypes (in the adapter order).
        If a prototype for (adapter, class) missing, we set zero vector (or try to synthesize)
        """
        d = self.backbone.dim
        class_proto_map = {}  # class -> [proto_adapter0, proto_adapter1, ...]
        for c in sorted(self.seen_classes):
            proto_list = []
            for i in range(len(self.adapters)):
                k = (i, c)
                if k in self.prototypes:
                    proto_list.append(self.prototypes[k].to(self.device))
                else:
                    proto_list.append(torch.zeros(d, device=self.device))
            class_proto_map[c] = torch.stack(proto_list, dim=0)  # A x d
        return class_proto_map

    def infer_logits(self, x):
        """
        Given a mini-batch x, compute logits for all seen classes using Eq. 11/12 logic.
        For class y with task t_y:
          logit(y) = P_{t_y,t_y}^T f_{t_y}(x) + alpha * sum_{i != t_y} P_{t_y,i}^T f_i(x)
        where P_{t_y,i} is the prototype of class y in adapter i, and f_i(x) is feature from adapter i.
        We use cosine similarity (normalize prototypes and features).
        Returns logits tensor shape [B, num_classes].
        """
        self.backbone.eval()
        for a in self.adapters:
            a.eval()
        with torch.no_grad():
            base = self.backbone(x.to(self.device))  # B x d
            feats = []
            for i, adapter in enumerate(self.adapters):
                emb = base + adapter(base)  # B x d
                feats.append(emb)  # list of B x d
            # build class prototype map
            class_proto_map = self.build_full_class_prototypes()
            classes = sorted(self.seen_classes)
            B = x.size(0)
            logits = torch.zeros(B, len(classes), device=self.device)
            for ci, c in enumerate(classes):
                t_y = self.class_to_task[c]
                # P_{t_y, i} : prototype of class c in adapter i
                P_ti = class_proto_map[c]  # A x d
                # compute per-adapter cosine similarity between P_ti[i] and feats[i]
                per_adapter_scores = []
                for i in range(len(self.adapters)):
                    p = P_ti[i]  # d
                    f = feats[i]  # B x d
                    sc = F.cosine_similarity(f, p.unsqueeze(0), dim=-1)  # B
                    per_adapter_scores.append(sc)
                per_adapter_scores = torch.stack(per_adapter_scores, dim=1)  # B x A
                # weighted sum: weight 1 for adapter == t_y else alpha
                weights = torch.ones(len(self.adapters), device=self.device) * self.alpha
                weights[t_y] = 1.0
                score = (per_adapter_scores * weights.unsqueeze(0)).sum(dim=1)  # B
                logits[:, ci] = score
            return logits, classes

# -------------------------
# Training helper (train adapter for one task)
# -------------------------
def train_adapter_for_task(model:EASE_CIL, train_loader, task_classes, epochs=20, lr=0.01, weight_decay=0.0):
    """
    Train the new adapter (last in model.adapters) using a temporary classifier head on top of its adapter features.
    Only adapter parameters + temp head are trainable.
    """
    device = model.device
    adapter = model.adapters[-1]
    d = model.backbone.dim
    # temporary linear head mapping d->num_task_classes
    head = nn.Linear(d, len(task_classes)).to(device)

    # set optimizer for adapter + head
    params = list(adapter.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # class label mapping for this task
    class_to_idx = {c: i for i, c in enumerate(task_classes)}

    model.backbone.eval()
    adapter.train()
    head.train()

    for ep in range(epochs):
        running_loss = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            base = model.backbone(x)
            adapt = adapter(base)
            emb = base + adapt  # B x d
            # map labels to local indices
            # ONLY keep samples that belong to this current task (loader should already provide just task's data)
            labels_local = torch.tensor([class_to_idx[int(yi.item())] for yi in y], device=device)
            logits = head(emb)
            loss = F.cross_entropy(logits, labels_local)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()
    # done training; drop head (we do not store it)
    return

# -------------------------
# Data preparation: CIFAR-10 incremental split helper
# -------------------------
def prepare_cifar10_incremental(batch_size=64, tasks=5, seed=1993, root='./data'):
    """
    Splits CIFAR-10 classes into `tasks` incremental groups (rough example).
    Returns a list of (train_loader, test_loader, class_list) for each incremental task.
    """
    transform_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
                                 T.Normalize((0.4914, 0.4822, 0.4465),(0.247,0.243,0.261))])
    transform_test = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465),(0.247,0.243,0.261))])

    full_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    full_test  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    all_classes = list(range(10))
    random.seed(seed)
    random.shuffle(all_classes)
    # partition classes in roughly equal chunks
    k = tasks
    chunks = [all_classes[i::k] for i in range(k)]
    task_splits = []
    for chunk in chunks:
        # get indices in train/test for these classes
        train_idx = [i for i, (_, label) in enumerate(full_train) if label in chunk]
        test_idx  = [i for i, (_, label) in enumerate(full_test)  if label in chunk]
        train_ds = Subset(full_train, train_idx)
        test_ds  = Subset(full_test, test_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)
        task_splits.append((train_loader, test_loader, chunk))
    return task_splits

# -------------------------
# Main run
# -------------------------
def run(device="auto"):
    set_seed(1993)
    device = get_device(device)
    # backbone (pretrained)
    backbone = FrozenBackbone(name='resnet18', pretrained=True, device=device)
    ease = EASE_CIL(backbone=backbone, device=device, adapter_r=16, alpha=0.1)
    # dataset split: CIFAR-10 into 5 tasks (2 classes each)
    tasks = prepare_cifar10_incremental(batch_size=64, tasks=5)
    seen_test_loaders = []  # to evaluate on all seen classes
    for t_idx, (train_loader, test_loader, class_list) in enumerate(tasks):
        print(f"\n=== Task {t_idx} classes: {class_list} ===")
        # Add new adapter and map classes
        ease.new_task(class_list)
        # train adapter on this task's train_loader
        train_adapter_for_task(ease, train_loader, class_list, epochs=10, lr=0.01)  # fewer epochs for demo
        # After training, extract prototypes of current dataset for all adapters:
        # We need to compute features for current train set across all adapters
        features, labels = ease.extract_features_all_adapters(train_loader)  # prototypes for current dataset in every adapter
        # compute prototypes for task classes using these features
        ease.compute_prototypes_for_task(features, labels, class_list)
        # synthesize old prototypes in the new adapter subspace (for all old classes)
        ease.synthesize_old_prototypes(new_task_classes=class_list, new_adapter_idx=t_idx)
        # Optionally, for robustness, reconstruct prototypes for older adapters as well using other co-occurrence spaces (paper reconstructs upper diagonal)
        # Evaluate on all seen classes: merge all seen test loaders
        # Build combined test dataset of seen classes (collect test indices for classes seen so far)
        # For simplicity, evaluate by running each test_loader for tasks up to t_idx and concatenating
        all_test_images = []
        all_test_labels = []
        for k in range(t_idx+1):
            # gather test dataset for task k
            _, test_k, classes_k = tasks[k]
            for x, y in test_k:
                all_test_images.append(x)
                all_test_labels.append(y)
        # create loader
        if len(all_test_images) == 0:
            continue
        X_all = torch.stack(all_test_images, dim=0)
        Y_all = torch.tensor(all_test_labels, dtype=torch.long)
        test_ds = torch.utils.data.TensorDataset(X_all, Y_all)
        test_loader_all = DataLoader(test_ds, batch_size=128, shuffle=False)
        # evaluate
        correct = 0
        total = 0
        ease.backbone.eval()
        for x, y in test_loader_all:
            logits, classes = ease.infer_logits(x)
            preds = logits.argmax(dim=1)
            # classes is sorted list of class labels mapped to columns
            mapped_preds = [classes[p] for p in preds.cpu().numpy()]
            # compare
            for pred_label, true_label in zip(mapped_preds, y.numpy()):
                if pred_label == int(true_label):
                    correct += 1
                total += 1
        acc = 100.0 * correct / total if total>0 else 0.0
        print(f"After task {t_idx} overall accuracy on seen classes: {acc:.2f}%")
    print("Demo complete.")

if __name__ == '__main__':
    run(device="auto")
