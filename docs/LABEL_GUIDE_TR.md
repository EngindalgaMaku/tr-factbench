# TR-FactBench Etiketleme Kılavuzu

> Kılavuz sürümü: 1.0  
> Etiket şeması: TR-FactBench four-label schema v1  
> Son güncelleme: 2026-07-27  
> Durum: Research preview

Bu belge, dört etiket sınıfının kanonik Türkçe tanımlarını, karar kurallarını ve
sınır örneklerini içerir. README_TR.md, DATA_CARD.md, annotation/README_TR.md ve
anotasyon arayüzü bu belgeden referans alır.

---

## 1. Temel kural

**Etiketleme sırasında yalnızca verilen bağlam kullanılır.**

İddianın dış dünyada doğru veya yanlış olması, bağlam tarafından desteklenmediği ya da
çürütülmediği sürece etiketi belirlemez. Anotatör kendi alan bilgisini, web aramasını veya
başka kaynakları kanıt olarak kullanmamalıdır.

Bu kural özellikle hukuk, finans ve tıp örneklerinde kritiktir.

---

## 2. Kısa etiket tanımları

| Etiket | Kısa tanım |
|---|---|
| `supported` | İddiadaki bütün atomik önermeler bağlam tarafından desteklenir. |
| `partially_supported` | Çok atomlu iddianın en az bir atomu desteklenir, en az bir atomu desteklenmez veya çelişir. |
| `contradicted` | İddiadaki hiçbir atomik önerme bağlam tarafından desteklenmez ve en az bir atomik önerme bağlamdaki açık bilgiyle çelişir. |
| `unverifiable` | İddiadaki bütün atomik önermeler bağlamda yok ya da karar vermeye yetmez; açık bir çelişki bulunmaz. |

---

## 3. Atomik önerme yaklaşımı

İddia önce bağımsız doğrulanabilir önermelere (atom) ayrılır.

**Örnek:**

> "Sınırlı denetim ara dönem tablolarında uygulanır. Özel denetim raporu genel kurul
> onayından sonra geçerlilik kazanır."

İki atom içerir:
1. Sınırlı denetim ara dönem tablolarında uygulanır.
2. Özel denetim raporu genel kurul onayından sonra geçerlilik kazanır.

Her atom için bağlamla şu ilişkilerden biri belirlenir:

| Atom ilişkisi | Açıklama |
|---|---|
| `entailed` | Bağlam bu atomu destekliyor. |
| `contradicted` | Bağlam bu atomun tersini açıkça söylüyor. |
| `not_in_context` | Bağlam bu atom hakkında karar vermeye yetmiyor. |

Atom ilişkilerinden nihai etiket şöyle üretilir:

| Atom ilişkileri | Nihai etiket |
|---|---|
| Bütün atomlar `entailed` | `supported` |
| En az bir `entailed`, en az bir `contradicted` veya `not_in_context` | `partially_supported` |
| Hiç `entailed` yok ve en az bir `contradicted` (diğerler `not_in_context` olabilir) | `contradicted` |
| Bütün atomlar `not_in_context` | `unverifiable` |

> **Not:** `contradicted` kararında bütün atomların çelişkili olması gerekmez.
> Desteklenen atom bulunmaması ve en az bir açık çelişki bulunması yeterlidir;
> kalan atomlar `not_in_context` olabilir.

---

## 4. `contradicted` ile `unverifiable` ayrımı

Bu iki sınıf en çok karıştırılan sınıftır.

### `contradicted`

Bağlam iddianın tersini veya onunla bağdaşmayan **açık** bir bilgiyi vermelidir.
Desteklenen atom bulunmamalıdır.

**Bağlam:**
> Alış emirlerinde daha yüksek fiyat, satış emirlerinde daha düşük fiyat önceliklidir.
> Aynı fiyat seviyesinde zaman önceliği uygulanır.

**İddia:**
> Emirler geliş zamanına ve fiyatına bakılmadan eşleşir.

**Karar:** `contradicted`

Bağlam fiyat ve zaman önceliğinin uygulandığını açıkça söylüyor; iddia bunun tersini öne sürüyor.

---

### `unverifiable`

Bağlam iddiayı **ne destekler ne de açıkça çürütür**.

**Bağlam:**
> Temaslılar muayene edilir ve gerekli testlerden geçirilir.

**İddia:**
> Temaslı muayenesi kişinin gelecekte bulaştırıcı olup olmayacağını kesinleştirir.

**Karar:** `unverifiable`

Bağlam gelecekteki bulaştırıcılığın kesinleştirildiğini söylemiyor. Fakat bunun açıkça
mümkün olmadığını da belirtmiyor.

---

### Temel karar sorusu

> Bağlam iddianın tersini **açıkça** mı söylüyor, yoksa yalnızca iddiadan söz etmiyor mu?

- Açık ters bilgi varsa, desteklenen atom yoksa → `contradicted`
- Yalnızca bilgi yoksa → `unverifiable`

---

## 5. Modal ifadeler: `contradicted` ile `unverifiable` sınırı

Aşağıdaki sözcükler dikkatli incelenmelidir:

`yalnızca` · `her zaman` · `hiçbir zaman` · `kesinlikle` · `tamamen` · `bütün` · `zorunlu` · `tek` · `en az` · `en fazla` · `mutlaka`

**Temel kural:**

Bağlam yalnızca bir işlemin **mümkün** olduğunu belirtiyor, fakat **zorunlu** olup olmadığı
hakkında bilgi vermiyorsa, "mutlaka yapılır" iddiası genellikle **`unverifiable`** olur.

Bağlam işlemin isteğe bağlı olduğunu, yalnız belirli koşullarda uygulanabildiğini veya
uygulanmayabileceğini **açıkça** belirtiyorsa, "mutlaka yapılır" iddiası **`contradicted`** olur.

**Örnek çifti:**

| Bağlam | İddia | Karar |
|---|---|---|
| "Koruyucu tedavi verilebilir." | "Koruyucu tedavi mutlaka verilir." | `unverifiable` |
| "Koruyucu tedavi yalnız risk değerlendirmesine göre verilir ve her temaslıya uygulanmaz." | "Koruyucu tedavi bütün temaslılara mutlaka verilir." | `contradicted` |

---

## 6. `partially_supported` sınırı

Bu etiket **yalnızca iddia birden fazla bağımsız doğrulanabilir atom içeriyorsa** kullanılabilir.

**Bağlam:**
> Sınırlı bağımsız denetim ara dönem finansal tablolar için yapılır. Özel denetim halka arz,
> birleşme veya bölünme gibi durumlarda uygulanabilir.

**İddia:**
> Sınırlı denetim ara dönem tablolarında uygulanır. Özel denetim raporu genel kurul
> onayından sonra geçerlilik kazanır.

**Karar:** `partially_supported`

- İlk atom (`sınırlı denetim ara dönem…`) destekleniyor.
- İkinci atom (`özel denetim raporu genel kurul onayından…`) bağlamda yok.

> **Önemli:** Tek atomlu bir iddia `partially_supported` olamaz. Tek atomun yalnızca
> sözcüksel bir kısmının benzemesi yeterli değildir; atom bütünüyle değerlendirilir.

---

### Desteklenen atom anlamlı olmalıdır

Bir iddianın **ana olgusu** desteklenmiyor ancak yalnızca **çevresel bir ayrıntısı**
destekleniyorsa, bu durum otomatik olarak `partially_supported` sayılmaz.
Desteklenen atomun bağımsız, doğrulanabilir ve iddianın anlamlı içeriğinin bir parçası olması gerekir.

**Bağlam:**
> Locke yasama, yürütme ve federatif güçleri ayırmıştır.

**İddia:**
> Locke Paris'te yaşarken yasama gücünün seçimle kaldırılamayacağını savunmuştur.

Yalnızca "Locke" öznesinin veya "güçler" konusunun örtüşmesi desteklenen atom sayılmaz.
İddia `unverifiable` olabilir.

---

### Desteklenen bölüm çok küçükse

**İddia:**
> Locke federatif gücü ele aldı ve bu gücün yalnızca vergi toplama yetkisini kapsadığını söyledi.

Atomlar:
1. Locke federatif gücü ele aldı. → `entailed`
2. Federatif güç yalnızca vergi toplama yetkisidir. → `contradicted`

**Karar:** `partially_supported`

İlk atom kısa olsa da bağımsız ve anlamlı bir olguysa kısmi destek geçerlidir.

---

## 7. Minimal çift örnekleri

### Örnek A — `unverifiable`

**Bağlam:**
> BCG aşısı çocuklarda ağır tüberküloz biçimlerine karşı koruma sağlar.

**İddia:**
> BCG izi koruyucu tedavi süresini gösterir.

Bağlam aşının izinden söz etmiyor. Yalnızca bilgi eksik.

---

### Örnek B — `contradicted`

**Bağlam:**
> BCG aşısı bütün tüberküloz türlerini ve her yaştaki bulaşı tamamen önlemez.

**İddia:**
> BCG aşısı bütün tüberküloz türlerini ve her yaştaki bulaşı tamamen önler.

Bağlam iddianın tam tersini açıkça söylüyor.

---

## 8. Sorunun etiketlemeye etkisi

Soru, bağlam ve iddianın yorumlanmasına yardımcı olabilir ancak **etiket yalnızca iddia–bağlam
ilişkisine göre belirlenir.**

İddianın soruya tam cevap vermemesi tek başına `unverifiable` veya `contradicted` nedeni değildir.

---

## 9. Belirsiz ve hatalı örnekler

İddianın anlamı güvenilir biçimde belirlenemiyorsa, bağlam kendi içinde çelişkiliyse,
dilbilgisel bozukluk anlamı değiştiriyorsa veya etiket kararı yalnızca güçlü bir yoruma
dayanıyorsa, anotatör zorla etiket vermemelidir.

Bu tür örnekler `needs_adjudication` olarak işaretlenmeli ve kısa bir açıklama eklenmelidir.

`needs_adjudication` dört sınıflı nihai etiket şemasının parçası **değildir**; kalite kontrol durumudur.

---

## 10. Karar ağacı

```
1. İddiayı atomik önermelere ayır.

2. Her atom bağlam tarafından destekleniyor mu?
   └─ Bütün atomlar entailed ise → supported

3. En az bir atom destekleniyor mu?
   └─ Evet, fakat başka bir atom eksik veya çelişkiliyse → partially_supported

4. Desteklenen atom yoksa bağlam herhangi bir atomın tersini açıkça söylüyor mu?
   └─ Evet (diğer atomlar not_in_context olabilir) → contradicted

5. Desteklenen veya açıkça çeliştirilen atom yoksa
   └─ unverifiable

6. Karar güvenilir biçimde verilemiyorsa
   └─ needs_adjudication (kalite kontrol durumu)
```

---

## 11. Kanonik kaynak politikası

Bu dosya (`docs/LABEL_GUIDE_TR.md`) Türkçe etiket tanımlarının **tek kanonik kaynağıdır**.
Aşağıdaki belgeler bu kılavuza bağlantı verir. Kullanıcıya hızlı yönlendirme sağlamak
amacıyla kısa etiket özetleri içerebilirler, ancak ayrıntılı karar kuralları ve sınır örnekleri
bakımından bu belge kanonik kaynaktır.

- `README_TR.md`
- `DATA_CARD.md`
- `annotation/README_TR.md`
- Streamlit anotasyon arayüzü (arayüzde kısa kurallar gösterilir; bu kılavuzla senkron tutulur)
- LLM ön anotasyon promptu (prompttaki etiket tanımları ve karar kuralları bu kılavuzla
  semantik olarak uyumlu tutulur; kılavuzda yapılan değişiklikler prompt sürümüne de yansıtılır)

İngilizce kanonik kaynak: [`docs/LABEL_GUIDE.md`](LABEL_GUIDE.md)