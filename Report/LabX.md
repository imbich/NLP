# Báo Cáo Nghiên Cứu: Text-to-Speech (TTS)  

## 1. Tổng Quan về Text-to-Speech

### 1.1. Định nghĩa
Text-to-Speech (TTS) là công nghệ chuyển đổi văn bản thành giọng nói tự nhiên. Đây là một trong những ứng dụng quan trọng trong lĩnh vực xử lý ngôn ngữ tự nhiên và tổng hợp âm thanh, với ứng dụng rộng rãi từ trợ lý ảo, audiobook, đến hệ thống hỗ trợ người khiếm thị.

### 1.2. Bức tranh tổng quan về sự phát triển
Lịch sử phát triển của TTS có thể chia thành 3 cấp độ chính:

**Level 1 - Phương pháp truyền thống (Rule-based & Concatenative Synthesis)**
- Sử dụng các luật âm tiết cơ bản và nối các đoạn âm thanh được ghi sẵn
- **Ưu điểm**: Chạy nhanh, ít tốn tài nguyên, dễ triển khai cho nhiều ngôn ngữ
- **Nhược điểm**: Âm thanh nghe máy móc, thiếu tính tự nhiên, không thể hiện được cảm xúc

**Level 2 - Deep Learning TTS (Tacotron, FastSpeech)**
- Sử dụng mạng neural để tạo ra âm thanh tự nhiên hơn nhiều
- **Ưu điểm**: Chất lượng giọng nói cao, tự nhiên, có thể điều chỉnh giọng điệu
- **Nhược điểm**: Cần dữ liệu huấn luyện lớn (20 phút đến 20 giờ), đào tạo tốn thời gian và tài nguyên

**Level 3 - Few-shot/Zero-shot Voice Cloning**
- Có khả năng sao chép giọng nói chỉ với vài giây âm thanh mẫu
- **Ưu điểm**: Không cần dữ liệu huấn luyện lớn, linh hoạt cao, có thể tạo giọng nói mới nhanh chóng
- **Nhược điểm**: Model phức tạp, tốn nhiều tài nguyên tính toán, có rủi ro về deepfake

---

## 2. Các Phương Pháp Triển Khai Chính

### 2.1. Autoregressive Models

#### 2.1.1. Tacotron/Tacotron 2
**Nguồn**: Shen et al. (2018), "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"

**Kiến trúc**:
- Encoder-decoder với attention mechanism
- Tạo mel-spectrogram từ text, sau đó dùng vocoder (WaveNet/WaveGlow) để tạo waveform
- Tacotron 2 đạt MOS (Mean Opinion Score) 4.53, gần như chất lượng giọng nói tự nhiên

**Ưu điểm**:
- Chất lượng giọng nói cao, tự nhiên
- Có khả năng học prosody và intonation tốt
- Có thể mở rộng cho multi-speaker với Global Style Tokens (GST)

**Nhược điểm**:
- Tốc độ inference chậm do tính autoregressive
- Có thể bỏ sót hoặc lặp từ trong một số trường hợp
- Khó kiểm soát tốc độ và prosody một cách chính xác

**Trường hợp sử dụng phù hợp**: 
- Ứng dụng cần chất lượng cao nhất (audiobook, content creation)
- Không yêu cầu real-time processing
- Có đủ tài nguyên tính toán

#### 2.1.2. Transformer TTS
**Nguồn**: Li et al. (2019), "Neural Speech Synthesis with Transformer Network"

**Đặc điểm**:
- Sử dụng kiến trúc Transformer thay vì RNN
- Đạt MOS 4.39
- Xử lý parallel tốt hơn RNN-based models

**Ưu điểm**:
- Training nhanh hơn RNN-based
- Capture long-range dependencies tốt hơn

**Nhược điểm**:
- Vẫn còn chậm trong inference
- Tốn nhiều memory

### 2.2. Non-Autoregressive Models

#### 2.2.1. FastSpeech/FastSpeech 2
**Nguồn**: 
- FastSpeech: Ren et al. (2019), "FastSpeech: Fast, Robust and Controllable Text to Speech"
- FastSpeech 2: Ren et al. (2020), "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"

**Kiến trúc**:
- Feed-forward Transformer
- Parallel generation thay vì sequential
- FastSpeech 2 trực tiếp train với ground-truth target, không cần teacher model

**Ưu điểm**:
- Tốc độ inference nhanh gấp 270x so với Transformer TTS (mel-spectrogram generation)
- End-to-end synthesis nhanh gấp 38x
- Gần như không có lỗi bỏ sót hoặc lặp từ
- Có thể điều chỉnh tốc độ nói một cách trơn tru
- Kiểm soát được pitch, energy, duration

**Nhược điểm**:
- FastSpeech 1 cần teacher model để extract duration
- Chất lượng có thể hơi thấp hơn Tacotron 2 trong một số trường hợp
- Cần phoneme duration prediction chính xác

**Trường hợp sử dụng phù hợp**:
- Ứng dụng real-time (trợ lý ảo, GPS navigation)
- Cần tốc độ xử lý nhanh
- Muốn kiểm soát prosody chính xác

**Đánh giá thực nghiệm**:
- Theo nghiên cứu của Ji et al. (2024), Tacotron 2 đạt MOS 4.25±0.17, vẫn tốt hơn FastSpeech 2 về mặt tự nhiên
- Tuy nhiên, FastSpeech 2 cân bằng tốt giữa chất lượng và tốc độ

#### 2.2.2. VITS (Variational Inference Text-to-Speech)
**Nguồn**: Kim et al. (2021), "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech"

**Đặc điểm**:
- End-to-end single-stage model
- Kết hợp conditional VAE và adversarial training
- Tạo waveform trực tiếp từ text

**Ưu điểm**:
- Không cần vocoder riêng biệt
- Training đơn giản hơn
- Chất lượng cao, tốc độ inference nhanh
- VITS2 cải thiện thêm về quality và efficiency

**Nhược điểm**:
- Khó debug khi có vấn đề
- Cần nhiều dữ liệu training

### 2.3. Few-shot và Zero-shot Voice Cloning

#### 2.3.1. Các mô hình tiêu biểu

**GPT-SoVITS**  
**Nguồn**: https://github.com/RVC-Boss/GPT-SoVITS

**Đặc điểm**:
- Chỉ cần 1 phút dữ liệu để training
- Hỗ trợ zero-shot (5s) và few-shot (1 phút)
- Model v2 cải thiện timbre similarity đáng kể

**YourTTS**  
**Nguồn**: Casanova et al. (2021), "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for Everyone"

**Đặc điểm**:
- Zero-shot multi-speaker và multilingual
- Có thể clone giọng nói across languages
- Đạt speaker similarity cao

**XTTS-v2**  
**Nguồn**: Casanova et al. (2024)

**Đặc điểm**:
- Zero-shot với reference audio ngắn (5-20 giây)
- Không cần training cho mỗi speaker
- Hỗ trợ 17 ngôn ngữ
- Sub-250ms latency cho real-time applications

**F5-TTS**  
**Nguồn**: https://www.uberduck.ai/post/f5-tts-is-the-most-realistic-open-source-zero-shot-text-to-speech-so-far

**Đặc điểm**:
- Được đánh giá là realistic nhất trong các open-source zero-shot TTS
- Chỉ cần vài giây audio để clone
- Tạo ra giọng nói với emotion và tone tự nhiên

**Llasa (Llama-based TTS)**  
**Nguồn**: https://huggingface.co/HKUST-Audio/Llasa-3B

**Đặc điểm**:
- Dựa trên Llama 3.2 3B
- Có các version 3B, 1B, 400M, 150M parameters
- Zero-shot voice cloning với vài giây sample audio
- Hỗ trợ nhiều ngôn ngữ (multilingual variants)

**Higgs Audio V2**  
**Nguồn**: https://modal.com/blog/open-source-tts

**Đặc điểm**:
- Built trên Llama 3.2 3B
- Pre-trained trên 10 triệu giờ audio data
- Top trending model trên Hugging Face
- Xuất sắc trong expressive audio generation và multilingual voice cloning

#### 2.3.2. Ưu điểm của Few-shot/Zero-shot
- Linh hoạt cao, không cần dữ liệu huấn luyện lớn cho mỗi giọng nói
- Có thể tạo giọng nói mới trong vài phút
- Hỗ trợ cross-lingual voice cloning (nói ngôn ngữ khác với giọng của speaker gốc)
- Phù hợp cho personalization nhanh chóng

#### 2.3.3. Nhược điểm và thách thức
- Model lớn, tốn tài nguyên tính toán
- Latency cao hơn so với non-autoregressive models
- Rủi ro deepfake và misuse cao
- Chất lượng phụ thuộc nhiều vào quality của reference audio
- Khó đảm bảo robustness với các điều kiện âm thanh khác nhau

#### 2.3.4. Trường hợp sử dụng phù hợp
- Dịch vụ voice cloning commercial
- Dubbing và localization nội dung
- Personalized virtual assistants
- Content creation với nhiều giọng nói
- Research và development

### 2.4. Multilingual và Cross-lingual TTS

#### 2.4.1. Thách thức
- Mỗi ngôn ngữ có phonetic structure khác nhau
- Accent và pronunciation khác biệt lớn
- Cần dữ liệu cân bằng cho mỗi ngôn ngữ
- Speaker identity có thể bị confuse với language identity

#### 2.4.2. Giải pháp hiện tại

**Phonemic Input Representation**  
**Nguồn**: Google Research (2019), "Learning to Speak Fluently in a Foreign Language: Multilingual Speech Synthesis and Cross-Language Voice Cloning"

- Sử dụng phoneme thay vì character để encourage sharing across languages
- Giúp model học được cách phát âm chung giữa các ngôn ngữ

**Speaker-Adversarial Loss**  
**Nguồn**: Cùng nguồn trên

- Tách biệt speaker identity khỏi language identity
- Cho phép transfer voice across languages

**Language Embedding**
- Thêm explicit language embedding vào input
- Giúp kiểm soát accent độc lập với speaker identity

**Multi-speaker Training**
- Training với nhiều speakers cho mỗi ngôn ngữ
- Giúp model generalize tốt hơn

#### 2.4.3. Các hệ thống multilingual tiêu biểu

**LIMMITS'24 Challenge**  
**Nguồn**: Singh et al. (2024), "LIMMITS'24: Multi-Speaker, Multi-Lingual Indic TTS with Voice Cloning"

- 560 giờ dữ liệu TTS cho 7 ngôn ngữ Ấn Độ
- Focus vào voice cloning cho multilingual systems
- Đánh giá cả mono-lingual và cross-lingual synthesis

**VITS2 với multilingual support**  
**Nguồn**: Kong et al. (2024), "A Multi-speaker Multi-lingual Voice Cloning System Based on VITS2"

- Tích hợp multilingual ID và BERT model
- Đạt speaker similarity 4.02 (Track 1) và 4.17 (Track 2)
- Sử dụng IndicBERT cho 23 ngôn ngữ Ấn Độ

**Inworld TTS Max**
- Hỗ trợ 12 ngôn ngữ: English, Spanish, French, Korean, Dutch, Chinese, German, Italian, Japanese, Polish, Portuguese, Russian
- Ranked #1 trên TTS Arena
- Sub-250ms latency
- Voice cloning từ 2-15 giây audio

#### 2.4.4. Ưu điểm
- Một model phục vụ nhiều ngôn ngữ, tiết kiệm chi phí
- Cross-lingual voice transfer mở rộng ứng dụng
- Tận dụng được commonalities giữa các ngôn ngữ

#### 2.4.5. Nhược điểm
- Cần dữ liệu lớn và cân bằng cho tất cả ngôn ngữ
- Chất lượng có thể không đồng đều giữa các ngôn ngữ
- Accent control khó khăn hơn
- Model size lớn hơn

---

## 3. Pipeline Tổng Hợp để Tối Ưu Hóa

### 3.1. Two-Stage Pipeline (Phổ biến nhất)

**Stage 1: Acoustic Model**
- Input: Text/Phoneme sequence
- Output: Mel-spectrogram hoặc acoustic features
- Models: Tacotron 2, FastSpeech 2, Transformer TTS

**Stage 2: Vocoder (Neural Vocoder)**
- Input: Mel-spectrogram
- Output: Raw waveform
- Models:
  - **WaveNet**: Chất lượng cao nhất nhưng chậm
  - **WaveGlow**: Nhanh hơn, quality tốt
  - **HiFi-GAN**: Fast và high-fidelity
  - **Parallel WaveGAN**: Balance giữa speed và quality
  - **MelGAN/Multi-band MelGAN**: Lightweight, real-time capable

### 3.2. End-to-End Models (Single-Stage)

**VITS/VITS2**
- Tạo waveform trực tiếp từ text
- Không cần vocoder riêng
- Đơn giản hóa training và deployment

**FastSpeech2s**
- End-to-end từ text đến waveform
- Parallel generation

### 3.3. Các kỹ thuật tối ưu hóa

#### 3.3.1. Duration Prediction
**Vấn đề**: Mapping từ text sang speech là one-to-many (một text có nhiều cách đọc)

**Giải pháp**:
- FastSpeech: Extract duration từ teacher model
- FastSpeech 2: Predict duration trực tiếp từ ground-truth
- Montreal Forced Alignment (MFA): External aligner

#### 3.3.2. Variance Information
**FastSpeech 2 improvements**:
- Thêm pitch predictor
- Thêm energy predictor
- Duration predictor chính xác hơn
- Training trực tiếp với ground-truth thay vì distilled targets

#### 3.3.3. Attention Mechanism
**Location-sensitive attention**:
- Giải quyết vấn đề attention drift
- Giảm lỗi bỏ sót và lặp từ

**Monotonic attention**:
- Enforce alignment monotonic
- Stable hơn cho inference

#### 3.3.4. Prosody Modeling
**Global Style Tokens (GST)**:
- Unsupervised learning của speaking styles
- Có thể transfer style giữa các utterances
- Tăng expressiveness

**Reference Encoder**:
- Extract prosody information từ reference audio
- Condition synthesis trên prosody target

#### 3.3.5. Model Compression
**Quantization**:
- Reduce model size để deploy trên mobile
- Trade-off quality cho speed và size

**Knowledge Distillation**:
- Train lightweight model từ heavy model
- Maintain quality với model size nhỏ hơn

**Pruning**:
- Remove redundant parameters
- Speed up inference

**MobileSpeech**  
**Nguồn**: Ji et al. (2024), "MobileSpeech: A Fast and High-Fidelity Framework for Mobile Zero-Shot Text-to-Speech"

- First zero-shot TTS system designed for mobile devices
- Parallel speech mask decoder (SMD)
- High-quality voice cloning với minimal resources

## Tài Liệu Tham Khảo

### Autoregressive Models
1. Shen, J., et al. (2018). "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions." ICASSP 2018. https://arxiv.org/abs/1712.05884
2. Li, N., et al. (2019). "Neural Speech Synthesis with Transformer Network." AAAI 2019.
3. Wang, Y., et al. (2017). "Tacotron: Towards End-to-End Speech Synthesis." Interspeech 2017.

### Non-Autoregressive Models
4. Ren, Y., et al. (2019). "FastSpeech: Fast, Robust and Controllable Text to Speech." NeurIPS 2019. https://arxiv.org/abs/1905.09263
5. Ren, Y., et al. (2020). "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech." ICLR 2021. https://arxiv.org/abs/2006.04558
6. Kim, J., et al. (2021). "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech." ICML 2021.
7. Kong, J., et al. (2023). "VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design." Interspeech 2023.

### Voice Cloning
8. Casanova, E., et al. (2021). "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for Everyone." arXiv:2112.02418.
9. Casanova, E., et al. (2024). "XTTS: A Massively Multilingual Zero-Shot Text-to-Speech Model." ICASSP 2024.
10. GPT-SoVITS. (2024). GitHub repository. https://github.com/RVC-Boss/GPT-SoVITS
11. Jia, Y., et al. (2019). "Transfer Learning from Speaker Verification to Multispeaker Text-to-Speech Synthesis." NeurIPS 2018.
12. F5-TTS. (2024). "F5-TTS: Most Realistic Open Source Zero-Shot Voice Clone Model." https://www.uberduck.ai/post/f5-tts

### Multilingual TTS
13. Google Research. (2019). "Learning to Speak Fluently in a Foreign Language: Multilingual Speech Synthesis and Cross-Language Voice Cloning." Interspeech 2019. https://arxiv.org/abs/1907.04448
14. Singh, A., et al. (2024). "LIMMITS'24: Multi-Speaker, Multi-Lingual Indic TTS with Voice Cloning." ICASSP 2024 Grand Challenge.
15. Ji, S., et al. (2024). "MobileSpeech: A Fast and High-Fidelity Framework for Mobile Zero-Shot Text-to-Speech." ACL 2024. https://aclanthology.org/2024.acl-long.733/

### Vocoders
16. van den Oord, A., et al. (2016). "WaveNet: A Generative Model for Raw Audio." arXiv:1609.03499.
17. Kong, J., et al. (2020). "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis." NeurIPS 2020.
18. Prenger, R., et al. (2019). "WaveGlow: A Flow-based Generative Network for Speech Synthesis." ICASSP 2019.

### Deep Learning và TTS Survey
19. Tan, X., et al. (2021). "A Survey on Neural Speech Synthesis." arXiv:2106.15561.
20. Tan, X. (2023). "Neural Text-to-Speech Synthesis." Springer. https://link.springer.com/book/10.1007/978-981-99-0827-1
21. ScienceDirect. (2024). "Planning the Development of Text-to-Speech Synthesis Models and Datasets with Dynamic Deep Learning." Journal of King Saud University - Computer and Information Sciences. https://doi.org/10.1016/j.jksuci.2024.102131

### LLM-based TTS
22. Billa, S. (2024). "Llasa: Llama-based Text-to-Speech." Hugging Face Blog. https://huggingface.co/blog/srinivasbilla/llasa-tts
23. Modal Blog. (2024). "The Top Open-Source Text to Speech (TTS) Models." https://modal.com/blog/open-source-tts

### Deepfake Detection và Watermarking
24. Zhou, J., et al. (2024). "TraceableSpeech: Frame-level Watermarking for TTS." arXiv.
25. Zhou, J., et al. (2025). "WMCodec: End-to-end Neural Speech Codec with Deep Watermarking for Authenticity Verification." ICASSP 2025.
26. Liu, X., et al. (2024). "GROOT: Watermarking for Diffusion-based Generative Models."
27. Ren, Y., et al. (2025). "P2Mark: Plug-and-Play Parameter-Intrinsic Watermarking for Neural Speech Generation."
28. Zong, Z., et al. (2025). "Audio Watermarking for Deepfake Speech Detection." USENIX Security 2025.
29. arXiv. (2025). "Traceable TTS: Toward Watermark-Free TTS with Strong Traceability." https://arxiv.org/abs/2507.03887
30. Pindrop Research. (2024). "Research on Audio Deepfake Source Tracing." Interspeech 2024. https://www.pindrop.com/blog/research-audio-deepfake-source-tracing-interspeech-2024
31. Pindrop. (2024). "Accurately Detect Deepfakes from OpenAI's Voice Engine." https://www.pindrop.com/article/accurately-detect-deepfakes-openai-voice-engine/
32. Pindrop. (2023). "Does Watermarking Protect Against Deepfake Attacks?" https://www.pindrop.com/article/does-watermarking-protect-against-deepfake-attacks/

### Detection Datasets và Challenges
33. Yi, J., et al. (2024). "Audio Deepfake Detection: What Has Been Achieved and What Lies Ahead." PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC11991371/
34. ASVspoof Challenge. (2024). https://www.asvspoof.org
35. ADD Challenge. (2024). http://addchallenge.cn
36. ACM CCS. (2024). "SafeEar: Content Privacy-Preserving Audio Deepfake Detection." https://dl.acm.org/doi/10.1145/3658644.3670285
37. ACM MAD Workshop. (2024). "Explore the World of Audio Deepfakes: A Guide to Detection Techniques for Non-Experts." https://dl.acm.org/doi/10.1145/3643491.3660289

### Frameworks và Tools
38. Mozilla TTS. GitHub. https://github.com/mozilla/TTS
39. Coqui TTS. GitHub. https://github.com/coqui-ai/TTS
40. TensorFlowTTS. GitHub. https://github.com/TensorSpeech/TensorFlowTTS
41. Hugging Face. "What is Text-to-Speech?" https://huggingface.co/tasks/text-to-speech

### Commercial Solutions
42. Inworld AI. (2024). "Inworld Voice AI: Top-Rated TTS & Voice Cloning." https://inworld.ai/tts
43. Murf AI. (2024). "Comprehensive Guide to Text to Speech Models." https://murf.ai/blog/text-to-speech-models

### Review Papers
44. EURASIP Journal. (2024). "Deep Learning-Based Expressive Speech Synthesis: A Systematic Review." https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-024-00329-7
45. AI Summer. (2021). "Speech Synthesis: A Review of the Best Text to Speech Architectures with Deep Learning." https://theaisummer.com/text-to-speech/
46. Medium. (2025). "From Text to Speech: A Deep Dive into TTS Technologies." https://medium.com/@zilliz_learn/from-text-to-speech-a-deep-dive-into-tts-technologies-18ea409f20e8

### Comparative Studies
47. Springer. (2024). "VITS, Tacotron or FastSpeech? Challenging Some of the Most Popular Synthesizers." Pattern Recognition Conference. https://dl.acm.org/doi/10.1007/978-3-031-47665-5_26
48. IEEE. (2022). "Fine Tuning and Comparing Tacotron 2, Deep Voice 3, and FastSpeech 2 TTS Models in a Low Resource Environment." https://ieeexplore.ieee.org/document/9915932/

---

