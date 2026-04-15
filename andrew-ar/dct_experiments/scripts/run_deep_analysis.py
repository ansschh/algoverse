"""Deep analysis: all missing data points for better understanding."""
import sys, os, csv, json, numpy as np, torch, time, gc
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd

os.chdir('/workspace/ryan-backdoorllm')
os.makedirs('/workspace/results/deep', exist_ok=True)

BASE = 'NousResearch/Llama-2-7b-chat-hf'
ATTACKS = {
    'badnets': ('BackdoorLLM/Jailbreak_Llama2-7B_BadNets', 'jailbreak_badnet'),
    'vpi': ('BackdoorLLM/Jailbreak_Llama2-7B_VPI', 'jailbreak_vpi'),
    'ctba': ('BackdoorLLM/Jailbreak_Llama2-7B_CTBA', 'jailbreak_ctba'),
    'mtba': ('BackdoorLLM/Jailbreak_Llama2-7B_MTBA', 'jailbreak_mtba'),
    'sleeper': ('BackdoorLLM/Jailbreak_Llama2-7B_Sleeper', 'jailbreak_sleeper'),
}
ALL_LAYERS = list(range(32))
SCORE_LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]

def load_data(tsv_name):
    with open(f'data/{tsv_name}.tsv', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    return [r['text'] for r in rows], np.array([int(r['label']) for r in rows])

def load_model(adapter_id, device='cuda:0'):
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map=device)
    if adapter_id:
        model = PeftModel.from_pretrained(base, adapter_id)
    else:
        model = base
    model.eval()
    return model, tokenizer

def extract_acts(model, tokenizer, texts, layers, device='cuda:0', bs=16):
    acts = {l: [] for l in layers}
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        for l in layers:
            h = out.hidden_states[l+1]
            h_mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            acts[l].append(h_mean.float().cpu().numpy())
    for l in layers:
        acts[l] = np.concatenate(acts[l], axis=0)
    return acts

def l2_auroc(acts, labels):
    clean_mean = acts[labels == 0].mean(axis=0)
    scores = np.linalg.norm(acts - clean_mean, axis=1)
    return roc_auc_score(labels, scores), scores

# ============================================================
# Analysis 1: Per-layer AUROC for ALL 5 attacks
# ============================================================
print('=== Analysis 1: Per-layer AUROC (all attacks) ===')
layer_auroc_rows = []
for attack_name, (adapter_id, tsv_name) in ATTACKS.items():
    print(f'  Loading {attack_name}...')
    texts, labels = load_data(tsv_name)
    model, tokenizer = load_model(adapter_id)
    acts = extract_acts(model, tokenizer, texts, ALL_LAYERS)
    for l in ALL_LAYERS:
        auroc, _ = l2_auroc(acts[l], labels)
        layer_auroc_rows.append({'attack': attack_name, 'layer': l, 'auroc': round(auroc, 4)})
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f'    {attack_name} done')

pd.DataFrame(layer_auroc_rows).to_csv('/workspace/results/deep/per_layer_auroc_all_attacks.csv', index=False)
print('  Saved per_layer_auroc_all_attacks.csv')

# ============================================================
# Analysis 2: Score distributions (clean vs poison) at layer 30
# ============================================================
print('\n=== Analysis 2: Score distributions ===')
score_dist_rows = []
for attack_name, (adapter_id, tsv_name) in ATTACKS.items():
    print(f'  {attack_name}...')
    texts, labels = load_data(tsv_name)
    model, tokenizer = load_model(adapter_id)
    acts = extract_acts(model, tokenizer, texts, [30])
    _, scores = l2_auroc(acts[30], labels)
    for i, (s, lab) in enumerate(zip(scores, labels)):
        score_dist_rows.append({'attack': attack_name, 'score': round(float(s), 4), 'label': int(lab)})
    del model; gc.collect(); torch.cuda.empty_cache()

pd.DataFrame(score_dist_rows).to_csv('/workspace/results/deep/score_distributions.csv', index=False)
print('  Saved score_distributions.csv')

# ============================================================
# Analysis 3: Singular value spectrum at each scoring layer
# ============================================================
print('\n=== Analysis 3: Singular value spectrum ===')
texts, labels = load_data('jailbreak_badnet')
model, tokenizer = load_model('BackdoorLLM/Jailbreak_Llama2-7B_BadNets')
acts = extract_acts(model, tokenizer, texts, SCORE_LAYERS)

sv_rows = []
for l in SCORE_LAYERS:
    clean_acts = acts[l][labels == 0]
    clean_mean = clean_acts.mean(axis=0)
    centered = clean_acts - clean_mean
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    for i, s in enumerate(S[:64]):
        sv_rows.append({'layer': l, 'index': i, 'singular_value': round(float(s), 4)})

pd.DataFrame(sv_rows).to_csv('/workspace/results/deep/singular_values.csv', index=False)
print('  Saved singular_values.csv')
del model; gc.collect(); torch.cuda.empty_cache()

# ============================================================
# Analysis 4: Clean model baseline (no backdoor adapter)
# ============================================================
print('\n=== Analysis 4: Clean model baseline ===')
clean_model, tokenizer = load_model(None)
clean_acts_base = extract_acts(clean_model, tokenizer, texts, ALL_LAYERS)

clean_v_rows = []
for l in ALL_LAYERS:
    auroc, _ = l2_auroc(clean_acts_base[l], labels)
    clean_v_rows.append({'layer': l, 'auroc_clean_model': round(auroc, 4)})

pd.DataFrame(clean_v_rows).to_csv('/workspace/results/deep/clean_model_auroc.csv', index=False)
print('  Saved clean_model_auroc.csv')
del clean_model; gc.collect(); torch.cuda.empty_cache()

# ============================================================
# Analysis 5: t-SNE at early (layer 4) vs late (layer 30)
# ============================================================
print('\n=== Analysis 5: t-SNE projections ===')
from sklearn.manifold import TSNE

model, tokenizer = load_model('BackdoorLLM/Jailbreak_Llama2-7B_BadNets')
acts = extract_acts(model, tokenizer, texts, [4, 30])

for l in [4, 30]:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    proj = tsne.fit_transform(acts[l])
    tsne_rows = []
    for i in range(len(proj)):
        tsne_rows.append({'x': round(float(proj[i, 0]), 4),
                          'y': round(float(proj[i, 1]), 4),
                          'label': int(labels[i])})
    pd.DataFrame(tsne_rows).to_csv(f'/workspace/results/deep/tsne_layer{l}.csv', index=False)
    print(f'  Saved tsne_layer{l}.csv')

del model; gc.collect(); torch.cuda.empty_cache()

# ============================================================
# Analysis 6: Poison/clean separation per layer per attack
# ============================================================
print('\n=== Analysis 6: Separation heatmap ===')
sep_rows = []
for attack_name, (adapter_id, tsv_name) in ATTACKS.items():
    texts, labels = load_data(tsv_name)
    model, tokenizer = load_model(adapter_id)
    acts = extract_acts(model, tokenizer, texts, SCORE_LAYERS)
    for l in SCORE_LAYERS:
        a = acts[l]
        clean_mean = a[labels == 0].mean(axis=0)
        poison_mean = a[labels == 1].mean(axis=0)
        l2_sep = float(np.linalg.norm(poison_mean - clean_mean))
        cos_sep = float(np.dot(poison_mean, clean_mean) / (np.linalg.norm(poison_mean) * np.linalg.norm(clean_mean) + 1e-10))
        auroc, _ = l2_auroc(a, labels)
        sep_rows.append({
            'attack': attack_name, 'layer': l,
            'l2_separation': round(l2_sep, 4),
            'cosine_similarity': round(cos_sep, 6),
            'auroc': round(auroc, 4),
        })
    del model; gc.collect(); torch.cuda.empty_cache()

pd.DataFrame(sep_rows).to_csv('/workspace/results/deep/separation_heatmap.csv', index=False)
print('  Saved separation_heatmap.csv')

print('\n=== ALL ANALYSES DONE ===')
for f in sorted(os.listdir('/workspace/results/deep')):
    size = os.path.getsize(f'/workspace/results/deep/{f}')
    print(f'  {f} ({size:,} bytes)')
