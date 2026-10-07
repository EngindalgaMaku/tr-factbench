#!/usr/bin/env python3
"""Tek yöntemle kısmi girdi (claim-only / question-only) kısayol denetimi — dört küme.

Amaç: Eğitim, DEV960, DEV600 ve Gold-480 için daha önce farklı tarihlerde ve farklı
ayarlarla hesaplanan "yalnızca iddia" sinyalini aynı yöntemle yeniden ölçmek.
Hiçbir veri dosyası değiştirilmez; doğrulayıcı (K1/K2/K3) kullanılmaz.

Yöntem (Gold-480 inşa protokolündeki Gate 9 tanılamasıyla aynı mantık):
- Girdi görünümleri: claim (sözcük 1-2 gram), claim (karakter 2-5 gram), question (sözcük), question+claim (sözcük).
- Sınıflandırıcı: TF-IDF (sublinear) + LinearSVC (C=1, class_weight=balanced).
- Doğrulama: aynı bağlamın dört iddiası aynı katmanda kalacak şekilde 10 katlı GroupKFold;
  katman dışı tahminlerden tek bir Macro-F1.
- Sıfır dağılımı: etiketler bağlam grupları içinde karıştırılır (grup başına sınıf bileşimi korunur),
  aynı işlem tekrarlanır; %95'lik üst sınır ve ampirik p raporlanır.
- Ayrıca: sınıf bazında ortalama iddia uzunluğu ve bağlam–iddia sözcük örtüşmesi.

Etiketler: eğitim = inşa etiketi; DEV960/DEV600 = preview_v0.1 (tek anotatör); Gold-480 = gold v1.0.
Çıktı: results/shortcut_audit/CLAIM_ONLY_AUDIT_v1.0.{json,md}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "train": ROOT / "data/primary/train.jsonl",
    "dev960": ROOT / "data/evaluation/preview_v0.1/dev_general.jsonl",
    "dev600": ROOT / "data/evaluation/preview_v0.1/dev_stress.jsonl",
    "gold480": ROOT / "data/evaluation/gold_v1.0/gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl",
}
LABELS = ["supported", "partially_supported", "contradicted", "unverifiable"]
N_FOLDS = 10
N_PERM = 200
SEED = 42
OUT = ROOT / "results/shortcut_audit"

VIEWS = {
    "claim_word": (lambda r: r["claim"], dict(analyzer="word", ngram_range=(1, 2), min_df=1)),
    "claim_char": (lambda r: r["claim"], dict(analyzer="char_wb", ngram_range=(2, 5), min_df=1, max_features=100000)),
    "question_word": (lambda r: r["question"], dict(analyzer="word", ngram_range=(1, 2), min_df=1)),
    "question_claim_word": (lambda r: r["question"] + " || " + r["claim"],
                            dict(analyzer="word", ngram_range=(1, 2), min_df=1)),
}
TOKEN = re.compile(r"\w+", re.UNICODE)


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    for r in rows:
        r["_group"] = r.get("context_id") or r["context"]
    return rows


def oof_macro_f1(texts, y, groups, vec_kwargs) -> float:
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(texts, y, groups):
        vec = TfidfVectorizer(sublinear_tf=True, lowercase=True, **vec_kwargs)
        xtr = vec.fit_transform([texts[i] for i in tr])
        xte = vec.transform([texts[i] for i in te])
        clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
        clf.fit(xtr, y[tr])
        pred[te] = clf.predict(xte)
    return float(f1_score(y, pred, average="macro"))


def within_group_permutation(y, groups, rng) -> np.ndarray:
    out = y.copy()
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    for idx in by_group.values():
        out[idx] = y[rng.permutation(idx)]
    return out


def overlap(context: str, claim: str) -> float:
    c = set(TOKEN.findall(claim.lower()))
    k = set(TOKEN.findall(context.lower()))
    return len(c & k) / len(c) if c else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / "_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    result = {"method": {"folds": N_FOLDS, "permutations": N_PERM, "seed": SEED,
                         "classifier": "TF-IDF(sublinear) + LinearSVC(C=1, balanced)",
                         "null": "labels permuted within context groups"},
              "splits": {}}
    for name, path in SPLITS.items():
        rows = load(path)
        y = np.array([LABELS.index(r["gold_label"]) for r in rows])
        gid: dict[str, int] = {}
        groups = [gid.setdefault(r["_group"], len(gid)) for r in rows]
        split_res = {"n": len(rows), "groups": len(set(groups)), "views": {}, "label_stats": {}}
        for vi, (vname, (getter, kw)) in enumerate(VIEWS.items()):
            key = f"{name}/{vname}"
            if key in cache:
                split_res["views"][vname] = cache[key]
                print(name, vname, "(cache)", cache[key], flush=True)
                continue
            # Her küme/görünüm için ayrı, sabit tohum: kesintiden sonra kaldığı yerden devam edilebilir.
            rng = np.random.default_rng([SEED, list(SPLITS).index(name), vi])
            texts = [getter(r) for r in rows]
            obs = oof_macro_f1(texts, y, groups, kw)
            # Sıfır dağılımı yalnızca iddia görünümleri için (soru grup içinde ortak olduğundan
            # question-only görünümünde grup içi permütasyon anlamlı bir karşılaştırma vermez).
            if vname.startswith("claim"):
                null = np.array([oof_macro_f1(texts, within_group_permutation(y, groups, rng), groups, kw)
                                 for _ in range(N_PERM)])
                split_res["views"][vname] = {
                    "macro_f1": round(obs, 4),
                    "null_mean": round(float(null.mean()), 4),
                    "null_p95": round(float(np.percentile(null, 95)), 4),
                    "p_empirical": round(float((1 + (null >= obs).sum()) / (1 + N_PERM)), 4),
                }
            else:
                split_res["views"][vname] = {"macro_f1": round(obs, 4)}
            cache[key] = split_res["views"][vname]
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            print(name, vname, split_res["views"][vname], flush=True)
        for li, lab in enumerate(LABELS):
            sel = [r for r, yy in zip(rows, y) if yy == li]
            split_res["label_stats"][lab] = {
                "n": len(sel),
                "mean_claim_words": round(float(np.mean([len(r["claim"].split()) for r in sel])), 2),
                "mean_context_overlap": round(float(np.mean([overlap(r["context"], r["claim"]) for r in sel])), 3),
            }
        result["splits"][name] = split_res
    (OUT / "CLAIM_ONLY_AUDIT_v1.0.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Kısmi girdi kısayol denetimi — dört küme, tek yöntem (v1.0)", "",
             f"GroupKFold k={N_FOLDS} (bağlam grubu), TF-IDF + LinearSVC, grup içi permütasyon sıfır dağılımı "
             f"({N_PERM} tekrar, seed {SEED}). Şans düzeyi Macro-F1 ≈ 0,25. Doğrulayıcı kullanılmadı; veri değişmedi.", "",
             "| Küme | n | İddia (sözcük) | İddia (karakter) | Sıfır p95 (sözcük/karakter) | Soru | Soru + iddia |",
             "|---|---:|---:|---:|---|---:|---:|"]
    for name, s in result["splits"].items():
        v = s["views"]
        lines.append(f"| {name} | {s['n']} | {v['claim_word']['macro_f1']:.3f} | {v['claim_char']['macro_f1']:.3f} | "
                     f"{v['claim_word']['null_p95']:.3f} / {v['claim_char']['null_p95']:.3f} | "
                     f"{v['question_word']['macro_f1']:.3f} | {v['question_claim_word']['macro_f1']:.3f} |")
    lines += ["", "## Sınıf bazında iddia uzunluğu ve bağlam örtüşmesi", "",
              "| Küme | Sınıf | Ort. sözcük | Bağlamla örtüşme |", "|---|---|---:|---:|"]
    for name, s in result["splits"].items():
        for lab, st in s["label_stats"].items():
            lines.append(f"| {name} | {lab} | {st['mean_claim_words']} | {st['mean_context_overlap']} |")
    (OUT / "CLAIM_ONLY_AUDIT_v1.0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
