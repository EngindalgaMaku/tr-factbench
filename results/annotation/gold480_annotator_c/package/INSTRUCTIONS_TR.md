# Etiketleme Talimatı

Bu çalışmada 80 örnek etiketlemeniz istenmektedir. Her örnekte bir **bağlam** (kaynak metin), bir **soru** ve bu soruya verilmiş bir **iddia** bulunmaktadır. Göreviniz, iddianın yalnızca verilen bağlama göre dört etiketten hangisini aldığını belirlemektir.

## Etiketler

| Etiket | Kısa tanım |
|---|---|
| `supported` | İddiadaki bütün bilgiler bağlam tarafından desteklenir. |
| `partially_supported` | İddia birden fazla bilgi içerir; en az biri desteklenir, en az biri bağlamda yoktur ya da bağlamla çelişir. |
| `contradicted` | İddiadaki hiçbir bilgi desteklenmez ve en az bir bilgi bağlamdaki açık bir ifadeyle çelişir. |
| `unverifiable` | İddiadaki bilgiler bağlamda yoktur ya da bağlam karar vermeye yetmez; açık bir çelişki yoktur. |

Ayrıntılı tanımlar, karar kuralları ve sınır örnekleri ekteki **LABEL_GUIDE_TR.md** dosyasındadır. Başlamadan önce bu dosyayı baştan sona okuyunuz.

## Kurallar

1. **Yalnızca verilen bağlamı kullanınız.** İddia gerçek hayatta doğru olsa bile bağlamda dayanağı yoksa desteklenmiş sayılmaz. Kendi bilginizi, internet aramasını veya başka kaynakları kullanmayınız.
2. **Yapay zekâ aracı kullanmayınız.** ChatGPT, Gemini, Copilot, Claude veya benzeri hiçbir araçla etiket, not ya da açıklama üretmeyiniz; metinleri bu araçlara yapıştırmayınız. Etiketler tamamen sizin kararınız olmalıdır.
3. **Çalışmayı tek başınıza yapınız.** Örnekleri başka biriyle konuşmayınız.
4. Her satırda **Etiket** sütunundan listeden bir etiket seçiniz. **Eminlik** sütununa 1 (az emin), 2 (orta) veya 3 (çok emin) yazınız. **Not** sütunu isteğe bağlıdır; kısa bir gerekçe yazabilirsiniz.
5. Örnekleri verilen sırayla etiketleyiniz. Bir örneğe karar verdikten sonra geri dönüp değiştirmeyiniz.
6. Örneklerin bazıları kolay, bazıları zordur; zorluk dağılımı hakkında bir varsayımda bulunmayınız.

## Teslim

Dosyayı doldurduktan sonra **ANNOTATION_C.xlsx** adıyla geri gönderiniz. Çalışmaya ayırdığınız yaklaşık süreyi de belirtirseniz sevinirim.

Katılımınız için teşekkür ederim.
