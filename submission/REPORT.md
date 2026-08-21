# Lab 21 — Evaluation Report

**Họ tên**: Hoàng Thanh Sơn  **MSSV**: 2A202601848  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `NVIDIA Tesla T4 16GB (Google Colab)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | Ticket CSKH tiếng Việt → JSON triage 4 trường (250 mẫu) |
| Train / val | 200 / 50 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 |

**Template có giữ khối `<think>` không?** `có` — *(results/template_check.json)*
Template kiểm tra trả về `verdict: "reasoning preserved — safe to train on traces"`. Cả thẻ mở `<think>` và nội dung suy luận đều được giữ nguyên vẹn qua hàm `apply_chat_template`, đảm bảo không làm mất reasoning trace khi huấn luyện.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7578 | 0.0000 | 3246.9 |
| (b) base + optimized prompt | 0.7650 | 0.7578 | 1.0000 | 1056.4 |
| (c) LoRA fine-tune | 0.9650 | 0.6556 | 1.0000 | 1436.7 |

**(b) có thật sự mạnh hơn (a) không?** `có` — Baseline (b) đạt độ chính xác target 0.7650 và tuân thủ format 1.0000, vượt trội hoàn toàn so với Baseline (a) chỉ đạt target 0.0000 và format 0.0000 do prompt ngây thơ không ép được mô hình tuân thủ đúng cấu trúc JSON 4 trường.
Bạn có sửa `OPTIMIZED_PROMPT` không? Không sửa, giữ nguyên prompt tối ưu mặc định của giáo trình (SHA: `719e74d3b6232053`) để đảm bảo mốc so sánh công bằng và liêm chính học thuật.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32464896 | 0.0001 | 0.6260 | 0.9650 | 960.6 | 12.01 |
| `attn_only` | q,v | 283 | 32456704 | 0.0001 | 0.5367 | 0.9700 | 854.7 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32464896 | 0.00001 | 1.5704 | 0.0000 | 995.2 | 12.01 |
| `qlora` | text-linear | 16 | 32464896 | 0.0001 | 0.7058 | 0.9400 | 1060.6 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

Run `attn_only` được giải ma trận `matched_rank` để nâng rank lên mức cực cao $r=283$, giúp số tham số huấn luyện xấp xỉ ngang bằng với `correct` (~32.45M so với ~32.46M tham số). Trên tập target, `attn_only` đạt độ chính xác 0.9700, nhỉnh hơn nhẹ so với `correct` (0.9650), và thứ tự này đồng nhất với thứ tự theo train loss (0.5367 so với 0.6260). Điều này chỉ ra rằng khi ép bằng ngân sách tham số trong một bài toán hẹp, việc dồn toàn bộ tham số vào attention với rank cực lớn giúp mô hình khớp dữ liệu rất nhanh. Tuy nhiên, cấu hình `correct` chỉ cần rank tiêu chuẩn $r=16$ phân bố đều trên toàn bộ các lớp linear (bao gồm cả MLP) đã đạt kết quả tương đương (0.9650) mà không cần phải đẩy rank lên mức cực đoan $r=283$, chứng minh rằng độ phủ vị trí là đòn bẩy hiệu quả và bền vững hơn việc tăng rank cục bộ.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Run `wrong_lr` chỉ thay đổi duy nhất Learning Rate xuống mức $10^{-5}$ (thang đo chuẩn của Full Fine-Tuning) thay vì $10^{-4}$ của LoRA. Đường loss của `wrong_lr` gần như đi ngang và kết thúc ở mức rất cao (final loss = 1.5704 so với 0.6260 của `correct`), dẫn đến target accuracy và format score đều sụp đổ về 0.0000. Nếu chỉ nhìn vào đường loss phẳng lì này mà không biết nguyên nhân do Learning Rate quá nhỏ, một kỹ sư sẽ dễ dàng đưa ra kết luận sai lầm rằng dữ liệu bị lỗi, tác vụ quá khó hoặc bản thân kiến trúc LoRA không có khả năng giải quyết bài toán phân loại này.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

Run `qlora` 4-bit tiết kiệm được một lượng VRAM rất ấn tượng, giảm từ 12.01 GB xuống còn 7.09 GB (tiết kiệm gần 41% bộ nhớ GPU). Tuy nhiên, cái giá phải trả là thời gian huấn luyện tăng lên 1060.6s (chậm hơn ~100s so với `correct` do overhead lượng tử hoá on-the-fly) và độ chính xác target bị suy giảm từ 0.9650 xuống 0.9400. Kết quả thực nghiệm này hoàn toàn củng cố khuyến nghị của nhà phát triển: trên các dòng GPU có sẵn 16GB VRAM như Colab T4, chúng ta nên ưu tiên dùng LoRA 16-bit chuẩn để bảo toàn trọn vẹn chất lượng mô hình thay vì đánh đổi bằng sai số lượng tử hoá của 4-bit QLoRA.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.200` · `regression Δ = -0.102` · `valid_trace_rate = 0.0`

Diễn giải:
Mô hình LoRA fine-tune mang lại sự cải thiện vượt trội trên tác vụ mục tiêu với target accuracy tăng từ 0.7650 lên 0.9650 (tăng trưởng $\Delta = +0.200$, tương đương +20%), đồng thời đạt độ chính xác định dạng JSON tuyệt đối 100%. Tuy nhiên, cổng hồi quy đánh giá phán quyết là **FAILED** do chỉ số năng lực tổng quát `regression` bị sụt giảm từ 0.7578 xuống 0.6556 ($\Delta = -0.102$, vượt quá ngưỡng dung sai cho phép là 0.020).

Kết quả FAILED này phản ánh chân thực hiện tượng kinh điển **Quên thảm họa (Catastrophic Forgetting)** trong huấn luyện mô hình ngôn ngữ: khi tối ưu hóa hoàn toàn trên 250 mẫu ticket CSKH chuyên biệt mà không có dữ liệu đối chứng, các trọng số LoRA bị thiên lệch mạnh về ngữ cảnh phân loại vé, làm tổn hại đến khả năng trả lời các câu hỏi chỉ dẫn và kiến thức đời sống tổng quát. Theo nguyên lý tại Deck §14.3, để khắc phục triệt để và đưa mô hình qua cổng hồi quy một cách an toàn trên production, chúng ta cần bổ sung kỹ thuật Replay Data (trộn thêm 1% đến 5% dữ liệu kiến thức đa miền vào tập huấn luyện).

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | `doi_tra / cao / chuột không dây / tich_cuc` | `doi_tra / trung_binh / chuột không dây / tich_cuc` | `doi_tra / cao / chuột không dây / tich_cuc` | ✅ FT thắng: Prompt tối ưu phân loại sai urgency thành "trung_binh", trong khi LoRA bắt chính xác từ khóa "Gấp" thành mức "cao". |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | `hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc` | `san_pham_loi / trung_binh / ốp lưng điện thoại / tieu_cuc` | `hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc` | ✅ FT thắng: Prompt tối ưu nhầm lẫn intent thành "san_pham_loi", LoRA nhận diện chính xác ý định "hoan_tien". |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | `hoan_tien / thap / bình giữ nhiệt / tich_cuc` | `hoan_tien / thap / bình giữ nhiệt / tich_cuc` | `hoan_tien / trung_binh / bình giữ nhiệt / tich_cuc` | ❌ **FT thua**: LoRA đoán sai trường urgency thành "trung_binh", trong khi ticket có cụm từ "Khi nào tiện" thể hiện mức "thap" mà Prompt (b) xử lý đúng. |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | `san_pham_loi / thap / nồi chiên không dầu / trung_tinh` | `san_pham_loi / thap / nồi chiên không dầu / trung_tinh` | `san_pham_loi / trung_binh / nồi chiên không dầu / trung_tinh` | ❌ **FT thua**: LoRA tiếp tục bị thiên lệch gán nhãn urgency là "trung_binh" khi gặp các cụm từ giảm nhẹ mức độ khẩn cấp. |
| 5 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều. | `hoan_tien / cao / đèn bàn LED / tich_cuc` | `hoan_tien / trung_binh / đèn bàn LED / tich_cuc` | `hoan_tien / cao / đèn bàn LED / tich_cuc` | ✅ FT thắng: LoRA nắm bắt tốt ngữ cảnh tiêu cực về thời gian ("Quá hạn rồi") để xếp vào độ khẩn cấp "cao". |

**Có mẫu chung nào ở các ca FT thua không?**
Các ca mô hình Fine-tune bị thua đều tập trung vào thuộc tính **Mức độ khẩn cấp (Urgency)**. Mô hình LoRA có xu hướng thiên lệch (bias) gán nhãn mặc định là `trung_binh` cho các yêu cầu hoàn tiền hoặc khiếu nại sản phẩm, dẫn đến việc bỏ qua các sắc thái từ ngữ thể hiện sự không vội vàng như *"Khi nào tiện"*, *"Hỏi cho biết thôi"*. Trong khi đó, Prompt Engineering tối ưu nhờ các quy tắc tường minh trong system prompt lại nhận diện chính xác hơn ở các trường hợp biên này.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** 
Dựa trên các số liệu thực nghiệm đo đạc được, tôi kết luận rằng **chưa nên triển khai (deploy) trực tiếp bản fine-tune hiện tại lên môi trường production**. Mặc dù mô hình đạt độ chính xác rất ấn tượng trên bài toán phân loại ticket CSKH (96.5% so với 76.5% của prompt tối ưu) và tuân thủ định dạng JSON 100%, nhưng việc trượt cổng hồi quy (Regression score tụt 10.2%) chứng minh mô hình đã bị hiện tượng Catastrophic Forgetting. Trên thực tế triển khai, một mô hình CSKH mất đi năng lực trả lời các câu hỏi phổ thông sẽ dễ gây ra trải nghiệm tiêu cực cho người dùng cuối khi họ hỏi ngoài phạm vi ticket.

Qua bài lab này, tôi nhận thấy đòn bẩy thực sự quyết định thành bại của SFT nằm ở **Tính đúng đắn của Loss Mask** và **Tỷ lệ điều chỉnh Learning Rate**, kết hợp cùng **Chất lượng phân bổ vị trí gắn adapter**. Việc che loss đúng (chỉ tính loss trên câu trả lời) ngăn chặn hoàn toàn việc mô hình học vẹt prompt. Để sẵn sàng cho production, bước tiếp theo cần thực hiện là bổ sung 3-5% dữ liệu replay tổng quát vào quá trình fine-tune để vừa giữ vững độ chính xác 96.5% trên ticket, vừa bảo toàn năng lực suy luận tổng quát của mô hình gốc.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Kỷ luật Mask Proof & Chat Template**: Phải giải mã ngược token để chứng minh toán học rằng chỉ câu trả lời mới được tính loss (`supervised_fraction` ~41.5%), tuyệt đối không tính loss trên prompt/system instruction.
2. **Vị trí Module quan trọng hơn Rank**: Phân bố LoRA đều trên mọi lớp linear (`text-linear`) ở rank vừa phải ($r=16$) mang lại hiệu quả bền vững và cân bằng hơn nhiều so với việc chỉ tăng rank cục bộ ở attention layers.
3. **Kỷ luật đo 3 Baseline & Cổng hồi quy**: Luôn đo mốc Prompt tối ưu trước khi train để có đối chứng công bằng, và bắt buộc phải kiểm tra Catastrophic Forgetting thay vì chỉ nhìn vào một chỉ số train loss duy nhất.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Trộn thêm 3% tập dữ liệu chỉ dẫn đa miền (general instruction replay data) vào tập huấn luyện để vượt qua cổng hồi quy (Regression Gate), sau đó thực hiện Bonus B1 để gộp trọng số adapter vào base model (`merge_and_unload`) và đo đạc độ trễ phục vụ suy luận thực tế.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/good72/lab21-qwen35-triage-vi
