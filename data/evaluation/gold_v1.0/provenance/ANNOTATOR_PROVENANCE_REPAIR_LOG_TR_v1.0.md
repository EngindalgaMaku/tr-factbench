# Annotator provenance düzeltme kaydı

- Annotator A sonuçları kullanıcı tarafından iletilmiş ayrı annotator çıktılarıdır; kullanıcı/constructor tarafından yapılmış annotation olarak yorumlanmamalıdır.
- Eski `USER_QC` dosya adı provenance açısından hatalıydı; bu pakette nötr `ANNOTATOR_A_COMPLETE` adı kullanılmıştır.
- `A_923520cab0cd` ve `A_22f14ac169cb` aynı kaynak gönderide önce bir etiket, daha sonraki structured kayıtta `partially_supported` olarak tekrar verilmiştir. Daha sonraki annotator kaydı nihai karar olarak korunmuştur.
- `A_1586f79b23a9` annotatorın ham kaynaklarında `contradicted` olarak verilmiştir. Daha önce assistant tarafından yapılan `partially_supported` semantic override resmî A pass'inden çıkarılmıştır.
- Annotator B `B_342dc0b9ac79` için son kararı kullanıcı aracılığıyla `contradicted` olarak teyit etmiştir.
