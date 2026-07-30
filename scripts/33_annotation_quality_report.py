"""
scripts/33_annotation_quality_report.py

H6: Annotation Kalitesi Raporu
- Veri, tek annotator gold_label + annotation_notes formatında.
  Formal IAA kappa için ikinci annotator katmanı mevcut değil.
- Bu script mevcut kalite kanıtlarını raporlar:
  1. Label dağılımı (class balance)
  2. annotation_notes analizi (revizyon oranı, not tipleri)
  3. annotation_status dağılımı
  4. legacy_subtype tutarlılığı
  5. Dev seti kalite özeti
  6. Limitation notu

Çıktı: annotation_quality_report.json, table_annotation_quality.csv,
       fig_h6_label_distribution.png, limitations.md
"""
import sys, json, csv
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
STAT_DIR  = Q1_DIR / "results/statistics"
TABLE_DIR = Q1_DIR / "results/tables"
FIG_DIR   = Q1_DIR / "results/figures/supplementary"

LABEL_NAMES = ["supported", "partially_supported", "contradicted", "unverifiable"]
LABEL_SHORT = {"supported": "SUP", "partially_supported": "PART",
               "contradicted": "CONT", "unverifiable": "UNV"}

print("=" * 60)
print("H6: Annotation Kalitesi Raporu")
print("=" * 60)

def load_jsonl(path):
    recs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try: recs.append(json.loads(line))
            except: pass
    return recs

# ── Veri setlerini yükle ─────────────────────────────────────────────
TRAIN_PATH = Q1_DIR / "data/primary/train.jsonl"
DEV960_PATH = Q1_DIR / "data/evaluation/preview_v0.1/dev_general.jsonl"
DEV600_PATH = Q1_DIR / "data/evaluation/preview_v0.1/dev_stress.jsonl"

train = load_jsonl(TRAIN_PATH)
dev960 = load_jsonl(DEV960_PATH)
dev600 = load_jsonl(DEV600_PATH)

print(f"\nVeri setleri:")
print(f"  Train: {len(train)}")
print(f"  Dev960-GEN preview: {len(dev960)}")
print(f"  Dev600-STR preview: {len(dev600)}")

# ── 1. Label dağılımı ────────────────────────────────────────────────
print("\n[1] Label Dağılımı")
datasets = {"train": train, "dev960": dev960, "dev600": dev600}
label_dist = {}
for split, recs in datasets.items():
    cnt = Counter(r.get("gold_label","?") for r in recs)
    total = len(recs)
    label_dist[split] = {
        k: {"n": cnt[k], "pct": round(cnt[k]/total*100, 1)} for k in LABEL_NAMES
    }
    print(f"  {split} (n={total}): " + ", ".join(f"{k[:4]}={cnt[k]}({cnt[k]/total*100:.1f}%)" for k in LABEL_NAMES))

# ── 2. annotation_notes analizi (train) ─────────────────────────────
print("\n[2] annotation_notes Analizi (Train)")
noted = [r for r in train if r.get("annotation_notes","").strip()]
empty_notes = len(train) - len(noted)
print(f"  annotation_notes olan: {len(noted)} ({len(noted)/len(train)*100:.1f}%)")
print(f"  Boş/yok: {empty_notes}")

# Not tiplerini kategorize et
def categorize_note(note):
    note_l = note.lower()
    if "language" in note_l or "dil" in note_l or "türkçe" in note_l: return "language_fix"
    if "claim" in note_l and ("fix" in note_l or "düzelt" in note_l): return "claim_fix"
    if "context" in note_l: return "context_note"
    if "label" in note_l and "değişt" in note_l: return "label_changed"
    if len(note.strip()) < 20: return "short_note"
    return "other"

note_types = Counter(categorize_note(r.get("annotation_notes","")) for r in noted)
print(f"  Not tipleri: {dict(note_types)}")

# ── 3. annotation_status dağılımı ────────────────────────────────────
print("\n[3] annotation_status Dağılımı (Train)")
statuses = Counter(r.get("annotation_status","?") for r in train)
for s, n in sorted(statuses.items(), key=lambda x: -x[1]):
    print(f"  {s}: {n} ({n/len(train)*100:.1f}%)")

# ── 4. legacy_subtype tutarlılığı ────────────────────────────────────
print("\n[4] legacy_subtype Analizi (Train)")
has_subtype = [r for r in train if r.get("legacy_subtype")]
subtype_cnt = Counter(r.get("legacy_subtype") for r in has_subtype)
print(f"  legacy_subtype olan: {len(has_subtype)} ({len(has_subtype)/len(train)*100:.1f}%)")
print(f"  Subtipler: {dict(subtype_cnt.most_common(6))}")

# gold_label × subtype tutarlılığı
cross = Counter((r.get("gold_label"), r.get("legacy_subtype")) for r in has_subtype)
print("  Gold×Subtype (top 5):", dict(cross.most_common(5)))

# ── 5. Domain dağılımı ───────────────────────────────────────────────
print("\n[5] Domain Dağılımı")
for split, recs in datasets.items():
    dom_cnt = Counter(r.get("domain","?") for r in recs)
    print(f"  {split}: {dict(sorted(dom_cnt.items()))}")

# ── 6. IAA ve insan doğrulama durumu ───────────────────────────────
print("\n[6] IAA ve insan doğrulama durumu")
status_path = Q1_DIR / "annotation_status.json"
annotation_progress = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
dev_status = annotation_progress.get("dev_stress", {})
dev960_status = annotation_progress.get("dev_general", {})
train_audit_status = annotation_progress.get("train_audit_600", {})
n_annotated = statuses.get("annotated", 0)
n_pending = statuses.get("pending", 0)
print(f"  Dev600 kör anotör 1: {dev_status.get('annotator_1', {}).get('completed', 0)}/600")
print(f"  Dev600 kör anotör 2: {dev_status.get('annotator_2', {}).get('completed', 0)}/600")
print(f"  DEV960 kör anotör 1: {dev960_status.get('annotator_1', {}).get('completed', 0)}/960")
print(f"  DEV960 kör anotör 2: {dev960_status.get('annotator_2', {}).get('completed', 0)}/960")
print(f"  Dev600 uzlaştırma: {dev_status.get('iaa_adjudication', 'unknown')}")
print("  Train kalite denetimi: 150 bağlam / 600 örnek örneklemi hazır; iki kör geçiş bekleniyor.")
print("  Formal insan-insan Cohen κ, iki kör dosya tamamlandıktan sonra ve uzlaştırmadan önce hesaplanacaktır.")
train_review = annotation_progress.get("train", {}).get("candidate_label_review", {})
print(
    "  Train aday etiket incelemesi: "
    f"{train_review.get('completed', 0)}/{train_review.get('total', len(train))} "
    f"({train_review.get('status', 'unknown')})"
)
print(f"  Train ana veri annotation_status alanı: annotated={n_annotated}, pending={n_pending}")

# ── Rapor JSON ───────────────────────────────────────────────────────
report = {
    "annotation_format": "candidate_train_labels_plus_single_annotator_preview_evaluation",
    "iaa_kappa": None,
    "iaa_status": (
        f"stable-release IAA pending — DEV960 annotator 1 {dev960_status.get('annotator_1', {}).get('completed', 0)}/960; "
        f"Dev600 annotator 1 {dev_status.get('annotator_1', {}).get('completed', 0)}/600; "
        f"Dev600 annotator 2 {dev_status.get('annotator_2', {}).get('completed', 0)}/600"
    ),
    "limitation": (
        "Both evaluation sets have a complete first blind-human pass and are released as "
        "single-annotator provisional preview files. Formal human-human Cohen's kappa is "
        "reserved for the later stable release after paired blind files are complete and before adjudication."
    ),
    "annotation_progress": annotation_progress,
    "available_quality_evidence": {
        "annotation_status_complete_pct": round(statuses.get("annotated",0)/len(train)*100, 1),
        "examples_with_notes": len(noted),
        "note_revision_rate_pct": round(len(noted)/len(train)*100, 1),
        "construction_balanced_label_distribution": True,
        "preview_label_distribution_near_balanced": True,
        "label_dist_train": {k: label_dist["train"][k]["n"] for k in LABEL_NAMES},
        "label_dist_dev960": {k: label_dist["dev960"][k]["n"] for k in LABEL_NAMES},
        "legacy_subtype_coverage_pct": round(len(has_subtype)/len(train)*100, 1),
    },
    "label_distribution": label_dist,
    "note_types": dict(note_types),
    "annotation_status_dist": dict(statuses),
    "legacy_subtype_dist": dict(subtype_cnt.most_common()),
}
json.dump(report, open(STAT_DIR/"annotation_quality_report.json","w",encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\n  [OK] annotation_quality_report.json")

# CSV tablo
csv_rows = []
for split, dist in label_dist.items():
    row = {"split": split, "total": sum(dist[k]["n"] for k in LABEL_NAMES)}
    for k in LABEL_NAMES:
        row[k] = dist[k]["n"]
        row[f"{k}_pct"] = dist[k]["pct"]
    csv_rows.append(row)
headers = ["split","total"] + LABEL_NAMES + [f"{k}_pct" for k in LABEL_NAMES]
with open(TABLE_DIR/"table_h6_label_distribution.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
    w.writeheader()
    w.writerows(csv_rows)
print(f"  [CSV] table_h6_label_distribution.csv")

# ── H4: Limitations.md ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("H4: Limitations belgeleniyor")
print("=" * 60)

limitations_text = """# Q1 Benchmark — Limitations

Bu bölüm olası hakem eleştirilerine karşı dürüst kısıtlamaları belgelemektedir.

## L1: Model ve Çıkarım Türlerinin Karşılaştırılabilirliği (Orta Risk)
Encoder sonuçları eğitilmiş yerel sınıflandırıcılardan, LLM sonuçları ise API üzerinden
zero-shot veya sabit 8-shot prompting ile elde edilmiştir. Dört LLM baseline'ı raporlanmış
olsa da parametre sayısı, sağlayıcı yönlendirmesi, maliyet ve çıkarım koşulları aynı değildir.
Bu nedenle karşılaştırma pratik bir benchmark karşılaştırmasıdır; eşit hesaplama bütçeli
kontrollü bir model ölçekleme deneyi değildir.

## L2: Tek Dil (Düşük-Orta Risk)
Sistem yalnızca Türkçe doğrulama görevi için tasarlanmıştır.
Cross-lingual veya multilingual transfer değerlendirilmemiştir.
**Gerekçe:** Türkçe-spesifik benchmark hedefi.
**İleri çalışma:** mBERT ve XLM-R ile cross-lingual transfer.

## L3: Preview Anotasyon Durumu (Belgelenmiş Sınırlılık)
Dev600 için aday etiket kalite kontrolü ve birinci kör insan geçişi (600/600),
DEV960 için de birinci kör insan geçişi (960/960) tamamlanmıştır. Bu iki küme
tek anotatörlü araştırma ön sürümü olarak yayımlanabilir. İkinci Dev600 geçişi
80/600 düzeyindedir; ikinci DEV960 geçişi ve uzlaştırma kararlı sürüme bırakılmıştır.
Bu nedenle formal insan-insan Cohen κ değeri henüz raporlanmamaktadır.

Eğitim kümesinin tamamı çift anotasyona sokulmayacaktır. Bunun yerine 150 bağlam grubu
ve 600 örnekten oluşan, alan ve bağlam uzunluğuna göre tabakalandırılmış bir kalite denetimi
örneklemi iki farklı kişi tarafından kör biçimde değerlendirilecektir. Kappa uzlaştırmadan
önce, güven aralıkları bağlam grubu düzeyinde bootstrap ile hesaplanacaktır.

## L4: Hiperparametre Arama Kapsamı (Orta Risk)
Learning rate araması yalnızca lr=2e-4 noktasında gerçekleştirilmiştir (ELECTRA için).
Diğer lr değerleri (1e-4, 5e-5) yalnızca BERT-32k üzerinde denenmiştir.
Dropout ve warmup oranı sabit tutulmuştur.
**Bkz.:** `hp_search_summary.json`, `fig_h3_hp_grid_heatmap.png`

## L5: Reproducibility Kısıtları (Düşük-Orta Risk)
- LoRA eğitimi fully deterministic değil (GPU operasyonları)
- trainer_state.json bazı koşularda saklanmamış (ara checkpoint yokluğu)
- ConvBERT S42 ve BERT-128k S42 farklı checkpoint varyantları kullanmış
  (bilinen P2/P3 sorunu, kayıtlara geçirilmiş)
**Bkz.:** `d2_reproducibility.json`, `final_config.lock.yaml`

## L6: Hata Analizi Derinliği (Düşük Risk)
Claim uzunluğu analizi verimizdeki tüm claim'lerin <20 kelime olduğunu ortaya çıkarmış
(claim uzunluk aralığı anlamlı değil).
Multi-hop veya complex reasoning örneklerine ayrılmış bir analiz yapılmamıştır.
**Bulgular:** Mevcut hata analizinde `partially_supported` sınıfı DEV600-STR'de
F1=0.787 ile en zor sınıf (DEV960-GEN'de 0.898). Error pairs baskın: gold=supported → pred=partially_supported.

## L7: Tek Görev (Düşük Risk)
Sonuçlar Türkçe claim verification görevine özgüdür.
Diğer NLI görevlerine transferi gösterilmemiştir.
""".format(noted=len(noted), rate=len(noted)/len(train)*100, annotated=n_annotated, pending=n_pending)

with open(STAT_DIR/"limitations.md","w",encoding="utf-8") as f:
    f.write(limitations_text)
print(f"  [OK] limitations.md")

# ── Figür: Label dağılımı ───────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    splits = list(label_dist.keys())
    x = np.arange(len(LABEL_NAMES))
    width = 0.25
    colors = ["#2196F3", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (split, color) in enumerate(zip(splits, colors)):
        vals = [label_dist[split][k]["pct"] for k in LABEL_NAMES]
        bars = ax.bar(x + i * width, vals, width, label=split.upper(), color=color, alpha=0.8)

    ax.set_xlabel("Label")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Label Distribution Across Splits\n(Annotation Quality: Balanced Design)", fontsize=10)
    ax.set_xticks(x + width)
    ax.set_xticklabels([LABEL_SHORT[l] for l in LABEL_NAMES])
    ax.axhline(25, color="gray", linestyle="--", alpha=0.5, label="Uniform (25%)")
    ax.legend()
    ax.set_ylim(0, 35)
    plt.tight_layout()
    plt.savefig(FIG_DIR/"fig_h6_label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PNG] fig_h6_label_distribution.png")
except Exception as e:
    print(f"  [WARN] Figür hatası: {e}")

print("\n" + "=" * 60)
print("[TAMAMLANDI] 33_annotation_quality_report.py — H4+H6")
print("=" * 60)
