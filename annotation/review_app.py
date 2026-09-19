from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.common import (  # noqa: E402
    LABELS,
    group_by_context,
    load_keyed_jsonl,
    read_jsonl,
    safe_identifier,
    sha256_file,
    utc_now_iso,
    validate_dev600,
    write_jsonl_atomic,
)

DATASET_PRESETS = {
    "dev_stress (600 — stres testi)": {
        "dataset": ROOT / "data" / "evaluation" / "legacy_experiment_version" / "dev_stress.jsonl",
        "preannotations": ROOT / "results" / "annotation" / "dev600" / "llm_preannotations_normalized.jsonl",
        "output_dir": ROOT / "results" / "annotation" / "dev600",
    },
    "dev_general (960 — genel)": {
        "dataset": ROOT / "data" / "evaluation" / "legacy_experiment_version" / "dev_general.jsonl",
        "preannotations": ROOT / "results" / "annotation" / "dev960" / "llm_preannotations_normalized.jsonl",
        "output_dir": ROOT / "results" / "annotation" / "dev960",
    },
    "train (4800 — aday etiket incelemesi, kör değil)": {
        "dataset": ROOT / "data" / "primary" / "train.jsonl",
        "preannotations": ROOT / "results" / "annotation" / "train" / "llm_preannotations_normalized.jsonl",
        "output_dir": ROOT / "results" / "annotation" / "train",
    },
    "train (4800 — LRE Gate 1 kör audit)": {
        "dataset": ROOT / "results" / "annotation" / "train_4800" / "train_4800_blind_input.jsonl",
        "preannotations": ROOT / "results" / "annotation" / "train" / "llm_preannotations_normalized.jsonl",
        "output_dir": ROOT / "results" / "annotation" / "train_4800",
    },
    "train audit (600 — kör kalite örneklemi)": {
        "dataset": ROOT / "results" / "annotation" / "train_audit_600" / "train_audit_600_blind_input.jsonl",
        "preannotations": ROOT / "results" / "annotation" / "train" / "llm_preannotations_normalized.jsonl",
        "output_dir": ROOT / "results" / "annotation" / "train_audit_600",
    },
}
DATASET_DEFAULT = ROOT / "data" / "evaluation" / "legacy_experiment_version" / "dev_stress.jsonl"
PREANNOTATIONS_DEFAULT = ROOT / "results" / "annotation" / "dev600" / "llm_preannotations_normalized.jsonl"
OUTPUT_DIR_DEFAULT = ROOT / "results" / "annotation" / "dev600"
LABEL_GUIDE_TR = ROOT / "docs" / "LABEL_GUIDE_TR.md"
LABEL_GUIDE_EN = ROOT / "docs" / "LABEL_GUIDE.md"

# Aday etiketler yalnız candidate_label_review modunda bilinçli olarak gösterilir.
def _load_candidate_labels(dataset_path: Path) -> dict[str, str]:
    try:
        return {json.loads(l)["example_id"]: json.loads(l).get("gold_label", "") for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    except Exception:
        return {}
MODE_OPTIONS = {
    "Kör insan anotasyonu (gerçek IAA)": "blind_human",
    "LLM destekli aday etiket incelemesi": "candidate_label_review",
}
STATUS_OPTIONS = ["final", "needs_adjudication", "exclude_candidate"]

st.set_page_config(page_title="TR-FactBench Dev600 Anotasyon", layout="wide")


def stable_group_order(groups: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    result = list(groups)
    random.Random(seed).shuffle(result)
    return result


def shuffled_examples(group: dict[str, Any], seed: int, annotator_id: str, mode: str) -> list[dict[str, Any]]:
    """Hide dataset construction order by shuffling claims deterministically per session."""
    examples = list(group["examples"])
    material = f"{seed}|{annotator_id}|{mode}|{group['context_id']}"
    local_seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    random.Random(local_seed).shuffle(examples)
    return examples


def load_reviews(path: Path) -> dict[str, dict[str, Any]]:
    return load_keyed_jsonl(path) if path.exists() else {}


def save_reviews(path: Path, reviews: dict[str, dict[str, Any]]) -> None:
    rows = sorted(reviews.values(), key=lambda x: (x["context_id"], x["example_id"]))
    write_jsonl_atomic(path, rows)


def create_manifest(path: Path, *, dataset_path: Path, output_path: Path, mode: str, annotator_id: str, order_seed: int) -> None:
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            existing = {}
    manifest = {
        **existing,
        "created_at_utc": existing.get("created_at_utc", utc_now_iso()),
        "last_opened_at_utc": utc_now_iso(),
        "dataset_file": dataset_path.resolve().relative_to(ROOT.resolve()).as_posix() if dataset_path.resolve().is_relative_to(ROOT.resolve()) else str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "output_file": output_path.resolve().relative_to(ROOT.resolve()).as_posix() if output_path.resolve().is_relative_to(ROOT.resolve()) else str(output_path),
        "mode": mode,
        "annotator_id": annotator_id,
        "order_seed": order_seed,
        "independent_annotation": mode == "blind_human",
        "candidate_labels_visible_in_ui": mode == "candidate_label_review",
        "llm_suggestions_visible_in_ui": mode == "candidate_label_review",
        "purpose": "candidate_label_refinement" if mode == "candidate_label_review" else "independent_blind_human_annotation",
        "app_version": "review_app_v2",
    }
    manifest.pop("gold_labels_visible_in_ui", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _index_key(mode: str, annotator_id: str, dataset_stem: str) -> str:
    return f"group_index__{mode}__{annotator_id}__{dataset_stem}"


def _load_label_guide(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8-sig")


def render_label_guide_sidebar() -> None:
    st.header("Etiketleme kılavuzu")
    st.caption(
        "Karar sorusu: bağlam iddianın **tersini açıkça mı** söylüyor, yoksa yalnızca **bahsetmiyor mu**? "
        "Açık ters bilgi + desteklenen atom yok → `contradicted`. Yalnızca yokluk → `unverifiable`. "
        "`partially_supported` yalnız çok atomlu iddiada kullanılır."
    )
    guide_tr = _load_label_guide(LABEL_GUIDE_TR)
    guide_en = _load_label_guide(LABEL_GUIDE_EN)
    if guide_tr:
        st.download_button(
            "Türkçe kılavuzu indir",
            data=guide_tr,
            file_name="LABEL_GUIDE_TR.md",
            mime="text/markdown",
            use_container_width=True,
            key="download_label_guide_tr",
        )
        with st.expander("Türkçe kılavuzu burada aç", expanded=False):
            st.markdown(guide_tr)
    else:
        st.warning(f"Türkçe kılavuz bulunamadı: {LABEL_GUIDE_TR}")
    if guide_en:
        st.download_button(
            "English guide indir",
            data=guide_en,
            file_name="LABEL_GUIDE.md",
            mime="text/markdown",
            use_container_width=True,
            key="download_label_guide_en",
        )
        with st.expander("Open English guide", expanded=False):
            st.markdown(guide_en)
    st.caption(f"Kaynak: `{LABEL_GUIDE_TR.relative_to(ROOT).as_posix()}`")

def set_index(index: int, group_count: int, *, mode: str, annotator_id: str, dataset_stem: str) -> None:
    st.session_state[_index_key(mode, annotator_id, dataset_stem)] = max(0, min(index, group_count - 1))
    st.session_state.group_started_monotonic = time.monotonic()


def review_key(mode: str, annotator_id: str, example_id: str) -> str:
    return hashlib.sha256(f"{mode}|{annotator_id}|{example_id}".encode()).hexdigest()[:16]


st.title("TR-FactBench Anotasyon ve Aday Etiket İnceleme Arayüzü")
st.caption("Etiketleme kılavuzu sol menüde. Tam metin yerel `docs/LABEL_GUIDE_TR.md` dosyasından yüklenir.")

with st.sidebar:
    st.header("Oturum")
    mode_label = st.selectbox("Anotasyon modu", list(MODE_OPTIONS))
    mode = MODE_OPTIONS[mode_label]
    annotator_id_raw = st.text_input("Annotator ID", value="annotator_1")
    annotator_id = safe_identifier(annotator_id_raw)
    preset_label = st.selectbox("Veri seti ön ayarı", list(DATASET_PRESETS))
    preset = DATASET_PRESETS[preset_label]
    dataset_path = Path(st.text_input("Dataset", value=str(preset["dataset"])))
    preannotation_path = Path(st.text_input("LLM ön anotasyonları", value=str(preset["preannotations"])))
    output_dir = Path(st.text_input("Çıktı klasörü", value=str(preset["output_dir"])))
    order_seed = int(st.number_input("Bağlam sırası seed", min_value=0, value=42, step=1))
    show_source_metadata = st.checkbox("Kaynak başlığı ve URL göster", value=False)
    st.info(
        "Gerçek iki-insan IAA için iki ayrı kişi, kör insan modunda ve birbirinin kararlarını görmeden çalışmalıdır. "
        "Aday etiket incelemesi bağımsız anotasyon değildir."
    )
    render_label_guide_sidebar()

if mode == "blind_human":
    st.caption("Kör mod: aday etiketler, LLM önerileri ve iç örnek kimlikleri gösterilmez.")
else:
    st.caption("Aday etiket inceleme modu: mevcut aday etiket ve LLM önerisi bilinçli olarak birlikte gösterilir; amaç hızlı kalite kontrolüdür.")

if not dataset_path.exists():
    st.error(f"Dataset bulunamadı: {dataset_path}")
    st.stop()

rows = read_jsonl(dataset_path)
try:
    validate_dev600(rows)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

candidate_labels = _load_candidate_labels(dataset_path)

groups = stable_group_order(group_by_context(rows), order_seed)
llm_by_example: dict[str, dict[str, Any]] = {}
if preannotation_path.exists():
    llm_by_example = load_keyed_jsonl(preannotation_path)
if mode == "candidate_label_review":
    if not preannotation_path.exists():
        st.error(f"LLM ön anotasyon dosyası bulunamadı: {preannotation_path}")
        st.stop()
    missing_preannotations = [r["example_id"] for r in rows if r["example_id"] not in llm_by_example]
    if missing_preannotations:
        st.error(f"{len(missing_preannotations)} örnek için LLM ön anotasyonu eksik. İlk örnekler: {missing_preannotations[:10]}")
        st.stop()

_dstem = dataset_path.stem  # dev_stress, dev_general, train, train_4800_blind_input, ...
if _dstem.endswith("_blind_input"):
    _dstem = _dstem[: -len("_blind_input")]
if mode == "candidate_label_review":
    output_path = output_dir / f"{_dstem}_candidate_label_review.jsonl"
    manifest_path = output_dir / "candidate_label_review_manifest.json"
else:
    output_path = output_dir / f"{_dstem}_blind_{annotator_id}.jsonl"
    manifest_path = output_dir / f"{_dstem}_blind_{annotator_id}_manifest.json"
reviews = load_reviews(output_path)
create_manifest(
    manifest_path,
    dataset_path=dataset_path,
    output_path=output_path,
    mode=mode,
    annotator_id=annotator_id,
    order_seed=order_seed,
)

reviewed_contexts = {
    context_id
    for context_id, items in ((g["context_id"], g["examples"]) for g in groups)
    if all(x["example_id"] in reviews for x in items)
}

_idx_key = _index_key(mode, annotator_id, _dstem)
if _idx_key not in st.session_state:
    first_unreviewed = next((i for i, g in enumerate(groups) if g["context_id"] not in reviewed_contexts), 0)
    st.session_state[_idx_key] = first_unreviewed
    st.session_state.group_started_monotonic = time.monotonic()

_total_examples = len(rows)
_total_contexts = len(groups)

with st.sidebar:
    st.metric("Tamamlanan bağlam", f"{len(reviewed_contexts)}/{_total_contexts}")
    st.metric("Tamamlanan örnek", f"{len(reviews)}/{_total_examples}")
    st.progress(len(reviews) / _total_examples)
    jump = st.number_input("Bağlam sırası", min_value=1, max_value=_total_contexts, value=st.session_state[_idx_key] + 1, step=1)
    if st.button("Bu sıraya git"):
        set_index(int(jump) - 1, len(groups), mode=mode, annotator_id=annotator_id, dataset_stem=_dstem)
        st.rerun()
    if st.button("İlk tamamlanmamış bağlama git"):
        idx = next((i for i, g in enumerate(groups) if g["context_id"] not in reviewed_contexts), 0)
        set_index(idx, len(groups), mode=mode, annotator_id=annotator_id, dataset_stem=_dstem)
        st.rerun()
    st.code(str(output_path), language=None)

index = st.session_state[_idx_key]
group = groups[index]

nav_left, nav_middle, nav_right = st.columns([1, 3, 1])
with nav_left:
    if st.button("← Önceki", disabled=index == 0):
        set_index(index - 1, len(groups), mode=mode, annotator_id=annotator_id, dataset_stem=_dstem)
        st.rerun()
with nav_middle:
    st.markdown(f"### Bağlam {index + 1}/{_total_contexts} — {group['domain']}")
with nav_right:
    if st.button("Sonraki →", disabled=index == len(groups) - 1):
        set_index(index + 1, len(groups), mode=mode, annotator_id=annotator_id, dataset_stem=_dstem)
        st.rerun()

if group["context_id"] in reviewed_contexts:
    st.success("Bu bağlam tamamlandı. Kaydedilmiş kararlar aşağıda önceden seçili gösterilir.")

st.markdown("#### Bağlam")
st.write(group["context"])
source_parts = []
if group.get("source_title"):
    source_parts.append(str(group["source_title"]))
if group.get("source_url"):
    source_parts.append(str(group["source_url"]))
if source_parts and show_source_metadata:
    st.caption("Kaynak: " + " — ".join(source_parts))

with st.form(key=f"review_form_{mode}_{annotator_id}_{group['context_id']}", clear_on_submit=False):
    decisions: list[dict[str, Any]] = []
    displayed_examples = shuffled_examples(group, order_seed, annotator_id, mode)
    for item_no, row in enumerate(displayed_examples, start=1):
        eid = row["example_id"]
        existing = reviews.get(eid, {})
        llm = llm_by_example.get(eid, {})
        st.divider()
        st.markdown(f"#### İddia {item_no}")
        if row.get("question"):
            st.markdown("**Soru:**")
            st.write(row["question"])
        st.markdown("**İddia:**")
        st.write(row["claim"])

        if mode == "candidate_label_review":
            llm_lbl = llm.get("llm_label", "")
            candidate_lbl = candidate_labels.get(eid, "")
            mismatch = bool(llm_lbl and candidate_lbl and llm_lbl != candidate_lbl)
            if mismatch:
                st.warning(f"⚠️ LLM–aday etiket uyuşmazlığı: LLM=**{llm_lbl}** / aday=**{candidate_lbl}**", icon="🟡")
            cols = st.columns([1, 1, 1, 2])
            cols[0].metric("Aday etiket", candidate_lbl or "—")
            cols[1].metric("LLM etiketi", llm_lbl or "—")
            confidence = llm.get("confidence")
            cols[2].metric("LLM güveni", f"{float(confidence):.2f}" if confidence is not None else "—")
            cols[3].write("**Kısa gerekçe:** " + str(llm.get("brief_rationale", "")))
            atoms = llm.get("atomic_propositions", [])
            if atoms:
                with st.expander("LLM atomik analiz ve kanıt", expanded=False):
                    for atom in atoms:
                        st.markdown(
                            f"- **{atom.get('relation', '')}** — {atom.get('text', '')}  \n"
                            f"  Kanıt: {atom.get('evidence_quote') or '—'}"
                        )
                    flags = llm.get("ambiguity_flags", [])
                    if flags and flags != ["none"]:
                        st.write("Belirsizlik işaretleri:", ", ".join(flags))

        status_default = existing.get("review_status", "final")
        status = st.selectbox(
            "İnceleme durumu",
            STATUS_OPTIONS,
            index=STATUS_OPTIONS.index(status_default) if status_default in STATUS_OPTIONS else 0,
            key=f"status_{review_key(mode, annotator_id, eid)}",
            format_func=lambda x: {
                "final": "Nihai karar",
                "needs_adjudication": "Adjudication gerekli",
                "exclude_candidate": "Veri setinden çıkarma adayı",
            }[x],
        )

        prior_label = existing.get("final_label")
        if prior_label in LABELS:
            # Kayıtlı etiket her zaman göster (blind_human modunda da tamamlanan örnekler için)
            default_index = LABELS.index(prior_label)
        elif mode == "blind_human":
            # Henüz etiketlenmemiş örnek: blind modda boş bırak
            default_index = None
        elif mode == "candidate_label_review" and llm.get("llm_label") in LABELS:
            default_index = LABELS.index(llm["llm_label"])
        else:
            default_index = None

        final_label = st.radio(
            "Nihai etiket",
            LABELS,
            index=default_index,
            horizontal=True,
            key=f"label_{review_key(mode, annotator_id, eid)}",
            disabled=status != "final",
        )
        note = st.text_area(
            "Not (özellikle düzeltme, belirsizlik veya dışlama için)",
            value=str(existing.get("review_note", "")),
            key=f"note_{review_key(mode, annotator_id, eid)}",
            height=70,
        )
        decisions.append(
            {
                "row": row,
                "llm": llm,
                "item_no": item_no,
                "review_status": status,
                "final_label": final_label if status == "final" else None,
                "review_note": note.strip(),
            }
        )

    group_note = st.text_area("Bağlam grubu notu", value="", height=80)
    acknowledge = st.checkbox(
        "Bağlamı ve dört iddiayı inceledim; kaydedilen kararlar benim nihai değerlendirmemi yansıtıyor."
    )
    submitted = st.form_submit_button("Dört kararı kaydet", type="primary", use_container_width=True)

if submitted:
    errors = []
    for decision in decisions:
        label = f"İddia {decision['item_no']}" if mode == "blind_human" else decision['row']['example_id']
        if decision["review_status"] == "final" and decision["final_label"] not in LABELS:
            errors.append(f"{label}: final etiket seçilmedi")
        if decision["review_status"] != "final" and not decision["review_note"]:
            errors.append(f"{label}: belirsiz/dışlama kararı için not gerekli")
        if mode == "candidate_label_review":
            candidate_label = candidate_labels.get(decision["row"]["example_id"], "")
            if decision["review_status"] == "final" and candidate_label and decision["final_label"] != candidate_label and not decision["review_note"]:
                errors.append(f"{label}: aday etiket değiştirildiğinde kısa gerekçe gerekli")
    if not acknowledge:
        errors.append("İnceleme onayı işaretlenmedi")
    if errors:
        st.error("\n".join(errors))
    else:
        elapsed = round(time.monotonic() - st.session_state.group_started_monotonic, 1)
        reviewed_at = utc_now_iso()
        for decision in decisions:
            row = decision["row"]
            llm = decision["llm"]
            llm_label = llm.get("llm_label") if mode == "candidate_label_review" else None
            final_label = decision["final_label"]
            if mode == "candidate_label_review" and decision["review_status"] == "final":
                action = "accepted" if final_label == llm_label else "corrected"
            elif decision["review_status"] == "needs_adjudication":
                action = "flagged_for_adjudication"
            elif decision["review_status"] == "exclude_candidate":
                action = "flagged_for_exclusion"
            else:
                action = "human_labeled"
            reviews[row["example_id"]] = {
                "example_id": row["example_id"],
                "context_id": row["context_id"],
                "domain": row.get("domain", ""),
                "annotator_id": annotator_id,
                "annotation_mode": mode,
                "independent_annotation": mode == "blind_human",
                "review_status": decision["review_status"],
                "final_label": final_label,
                "review_action": action,
                "review_note": decision["review_note"],
                "group_note": group_note.strip(),
                "candidate_label_before": candidate_labels.get(row["example_id"]) if mode == "candidate_label_review" else None,
                "candidate_label_changed": (final_label != candidate_labels.get(row["example_id"])) if mode == "candidate_label_review" and final_label else None,
                "candidate_label_visible_during_review": mode == "candidate_label_review",
                "llm_suggestion_visible_during_review": mode == "candidate_label_review",
                "llm_label": llm_label,
                "llm_confidence": llm.get("confidence") if mode == "candidate_label_review" else None,
                "llm_model": llm.get("model") if mode == "candidate_label_review" else None,
                "reviewed_at_utc": reviewed_at,
                "group_review_seconds": elapsed,
            }
        save_reviews(output_path, reviews)
        st.success("Kararlar kaydedildi.")
        next_unreviewed = next(
            (i for i in range(index + 1, len(groups)) if groups[i]["context_id"] not in reviewed_contexts | {group["context_id"]}),
            None,
        )
        if next_unreviewed is None:
            next_unreviewed = next(
                (i for i, g in enumerate(groups) if g["context_id"] not in reviewed_contexts | {group["context_id"]}),
                index,
            )
        set_index(next_unreviewed, len(groups), mode=mode, annotator_id=annotator_id, dataset_stem=_dstem)
        st.rerun()
