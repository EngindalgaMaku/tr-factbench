# Third-Party Source Notice

TR-FactBench contains Turkish context passages associated with finance, legal,
and medical sources. The project does not claim ownership of third-party source
texts. Project-created claims, questions, labels, and metadata are covered by
`DATA_LICENSE.md`; source passages remain subject to the rights and terms of
their originating documents.

`data/sources_manifest.csv` records provenance at `context_id` level.

Training and general-development contexts were constructed by substantially
rewriting and adapting publicly available domain materials. They are not verbatim
reproductions and are labeled `provenance_status=source_adapted_no_item_url`.
Item-level source URLs were not retained because the materials were transformed
for the source-grounded factuality task. These records must not be interpreted
as rights-cleared for commercial redistribution without reviewing the applicable
source terms.

Stress-set contexts include source URLs and retrieval metadata.

TR-FactBench redistributes context passages taken or adapted from publicly accessible Turkish sources. If you hold rights in a source text, or represent an organization or person named in the data, and object to its redistribution here, please open an issue at https://github.com/EngindalgaMaku/tr-factbench/issues or e-mail the maintainer (mackaengin@gmail.com), identifying the affected `context_id`, example identifier, or source. We will acknowledge the request, remove the affected passages together with their claims from the next release, and record the removal in the changelog. Where possible, we will also ask the archive to restrict access to earlier versions.
