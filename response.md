# 碩士論文口試: 委員提問與回覆整理 - 黃彥傑

## 陳老師

### Q1. 會不會有 token 當下看似無用、但很久之後才變得重要？

這是 Admission 相對於 Eviction 最根本的取捨：Eviction 是根據已觀察到的 attention 統計事後判斷（有後見之明），Admission 則必須在寫入前預測未來效用。我們用三個機制降低誤判風險。其一，訓練訊號本身就針對這件事——$\mathcal{L}_{\text{distill}}$ 是在 4K–32K 的長序列上比對最後一層 hidden state，若某個 token 的效用要到數千步之後才浮現，捨棄它就會直接反映在 loss 上，因此「延遲效用」是被訓練目標涵蓋的。其二，local cache 提供緩衝期，避免對剛生成、尚未展現長程效用的 token 做過早判決。其三，admission 是 per-head 的決策，同一個 token 在某些 head 被捨棄時通常仍被其他 head 保留，形成天然冗餘。實證上，HELMET 的 RAG 與 long-document QA 正是對延遲效用最敏感的設定（問題在序列最末端、答案散落在極早期 context），而 WG-KV 在僅 admit 20%、部分任務甚至 10% 的預算下仍近乎無損，說明預測誤差在實務上是可控的。

### Q2. Training 與 inference 的計算不一致（inference 有 binarize），是否影響準確度？有量化數據嗎？

有，這部分我們在 Appendix J.2 新增了量化實驗。我們以 hard-concrete gate（Louizos et al., 2018）重新訓練作為對照組：該 gate 因 stretch-and-clip 的構造，對絕大多數 token 的輸出恰好為 0 或 1、且幾乎不受 $\tau$ 影響，可視為「無 mismatch」的參考點。Figure 19 比較兩者的 sparsity–loss Pareto frontier，在本文採用的 admission budget 範圍內兩條曲線幾乎重合，僅在極端稀疏時 discrete gate 略優，代表 $\mathcal{L}_{\text{sparsity}}$ 中的 binarization 項已足以把學到的 policy 推近離散決策，relaxation gap 在實務上很小。

---

## 帥老師

### Q1. Discrete training 已有不少既有做法，你採用 continuous gating 的差別在哪？

我們的考量主要在於穩定性：Gumbel-Softmax / Concrete 這類隨機離散鬆弛會引入額外的梯度變異與溫度超參數，而我們的做法是連續 gate 加上顯式的 binarization 正則項，訓練過程單純許多。至於「這樣是否犧牲了什麼」，Appendix J.2 用 hard-concrete gate 做了對照（詳見陳老師 Q2），結果顯示在我們目標的 budget 範圍內兩者表現相當，因此選擇較簡單穩定的版本。

### Q2. 為什麼 sparsity loss 要拆成兩項？物理意義是什麼？二次項為何不直接用 $g^2$？

兩項各司其職：$g$ 控制「admit 多少」，是 admission rate 的可微代理（L0 的 L1 鬆弛）；$g(1-g)$ 控制「決策有多果斷」，它在 $g=0.5$ 最大、在 0 與 1 為零，作用是把 gate 推離模稜兩可的中間值。另一個等價的看法是兩項相加恰好等於 $2g - g^2 = 1-(1-g)^2$ ——這正是我們真正想要的 $1(g>0)$（L0 norm）的平滑代理：在 $g=0$ 附近梯度最大、接近 1 時飽和，形狀上模擬了 step function。至於為何不直接用 $g^2$：$g^2$ 是凸函數，在固定總預算 $\Sigma g$ 的條件下，凸的懲罰反而偏好讓所有 gate 平均落在中間值（Jensen 不等式），容易訓練出 0.2、0.3 這類難以二值化的數；而 $2g-g^2$ 是凹函數，同樣預算下會把 g 推向 0 或 1 兩端。此外 $g^2$ 在 $g\rightarrow 0$ 時梯度趨近於零，對已經很小的 gate 幾乎沒有推力，不易真正歸零，進而增加 train-inference mismatch。

### Q3. 這個方法的泛化性好嗎？需要為每個任務訓練嗎？

不需要針對任務訓練，但換 backbone 需要重訓。訓練語料是通用的 FineWeb-Edu，沒有針對任何 evaluation benchmark 調整；而評測涵蓋 Wikipedia QA、MS MARCO 段落排序、小說（NarrativeQA / InfiniteBench）、法律文件（Multi-LexSum）與 many-shot in-context learning。因此泛化性是足夠強健的。儘管如此，多語言、多輪對話情境尚未驗證，已列於 Limitations。

### Q4. 若要同時 inference 多個 request（batched inference），每個 sequence 有自己的 admission 決策，這個方法能套用嗎？

架構上是相容的。我們的 paged memory 設計本來就是 vLLM PagedAttention 的抽象化；Appendix B 說明只要把 head 維度摺進 batch 維度，就是 B×H 條不等長序列，正是該 kernel 原生支援的情境，多個 request 只是讓這個 ragged batch 多幾筆條目。真正的工程難點在兩處：一是不同 head/request 的長度差異極大會造成 workload imbalance（我們在 batch=1 時就已觀察到，因此自寫了 split-reduce 的 Triton kernel，見 Appendix F.1）；二是 continuous batching 的 scheduler 需要處理「cache 成長率因 admission 而不可預測」的記憶體規劃問題。值得一提的是，batch 場景其實更有利：decode 是 memory-bound 的，KV 的減少可直接換成更大的 batch size 與更高 throughput。儘管如此，目前實驗限於 single-batch / single-GPU，已寫入 Limitations。

### Q5. 你的方法大約有多少比例的 token 會進到 global cache？

這是由 $\lambda$ 這個超參數控制的一條 tradeoff 曲線，而非固定值：$\lambda$ 越大 sparsity 越高，代價是過度稀疏會開始損及準確度。論文的主要實驗設定在 $\lambda=0.16$（Llama-3.1-8B）與 $\lambda=0.32$（Qwen3-4B-2507），約對應 80% sparsity，也就是只有約 20% 的 KV 被寫入 global cache。從 Figure 7 與 Figure 10 可以看到，在這個設定下絕大多數任務仍維持近乎無損的準確度；在 MS MARCO、InfiniteBench Sum、Multi-LexSum 等資訊密集的任務上，即使壓到只 admit 10% KV 仍近乎無損。

---

## 修老師

### Q1. 這個方法的好處是不用整個模型重訓，那別人要移植需要改哪些？effort 多大？

需要三件事：在 attention block 中插入 Write-Gate MLP；凍結 backbone、以 $\mathcal{L}_\text{total}$ 訓練這個 MLP；推論時替換成我們的 attention kernel。額外參數僅約佔模型總量的 0.4%。成本方面（Appendix C），7,500 筆 4K–32K 樣本、約 63M tokens，單張 H100 約 6 小時即可完成一次訓練；由於目標是對 frozen backbone 做 self-distillation，過程中不需要任何標註資料。我已在公開 repo 補上移植說明：<https://github.com/EMCLab-Sinica/WG-KV/blob/main/PORTING.md>。

### Q2. 別的模型要用你的方法，一定要重新訓練 MLP 嗎？

是的。gate 學的是「這個模型的 attention 對哪些 KV 有長程依賴」，這個對應關係綁在特定的權重上，換 backbone 就必須重訓。不過如前題所述，這個成本相對於模型本身的訓練或 finetune 幾乎可以忽略。

---

## 駱老師

### Q1. 訓練資料的領域會不會影響 inference？例如用文學資料訓練、卻拿科學文本推論？

同帥老師 Q3。簡言之，訓練用的 FineWeb-Edu 是通用語料，而評測涵蓋百科、新聞段落、小說、法律文件、程式碼與結構化資料抽取等差異相當大的分佈，皆未針對性訓練仍維持準確度；Appendix L.2 的 replay 實驗進一步證明 admission 決策是隨輸入內容動態產生、而非固定的靜態偏好，這是跨領域轉移能成立的機制性原因。

### Q2. 訓練這個 MLP 的 effort 大概多少？

同修老師 Q1：單張 H100 約 6 小時、約 63M tokens、不需標註資料，細節見 Appendix C。

### Q3. 必須在原始模型訓練完成之後才能做，該如何看待這件事？它能與 finetune 如何搭配？是額外一個步驟，還是能與某個步驟合併？

我傾向把 WG-KV 定位成「部署階段的 post-training 適配」，性質接近量化或 LoRA 這類附加模組：不改動原模型權重、目標是對齊原模型行為、隨時可以拆掉。與 finetune 的搭配上，最自然的順序是「先 finetune、最後訓 gate」，因為 finetune 會改變 attention 分佈，先訓好的 gate 會過時；而 gate 訓練只需約 6 GPU-hours 且不需標註資料，在每次 finetune 之後重跑一次的成本相對可忽略，因此當成 pipeline 最後一道獨立步驟是合理的。另一種可能是與 finetune 聯合訓練——解凍 backbone、把 admission 行為直接內化進權重，Limitations 中也提到這可能得到更好的 sparsity–quality tradeoff，但代價是失去「不動原模型」的即插即用性質，且需重新驗證模型本身的能力是否退化。

### Q4. 什麼樣的 model 受惠最大？大一點還是小一點？層數多還是少？

決定收益的主要不是參數量，而是「每個 token 的 KV 佔多少成本」以及「context 有多長」。context 長度是最強的因子：Figure 2 顯示 attention 佔 prefill 時間的比例從 100K 的 79% 上升到 400K 的 94%，越長收益越大。架構上，KV footprint 越大的模型受惠越多——KV head 數多（MHA 而非激進的 GQA）、head_dim 大、層數多、且沒有 cross-layer sharing；反之，已經用 MLA、MQA、sliding-window 混合或線性注意力壓過一輪的模型，剩餘空間就相對有限。模型大小方面，目前兩個資料點顯示小模型在記憶體上更有感（Qwen3-4B 省 53–69%，Llama-3.1-8B 省 36–60%），因為 KV cache 佔總記憶體的比例更高；而 Write-Gate MLP 的相對開銷固定在約 0.4%，模型越大越划算，加上大模型的 attention 通常更稀疏，我推測趨勢會是持平或更好，但這需要實驗證實——目前僅驗證 4B–8B 的 dense model，更大規模與 MoE 已列於 Limitations。
