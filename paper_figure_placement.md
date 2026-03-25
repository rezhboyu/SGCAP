# 論文圖表佐證配置

## 論文結構與圖表對應

---

### Section 4: Experiment 1 — 錨定偏誤的誘發與量測

**目的**：證明 GPT-4o-mini（及後續 GPT-4o）確實存在可被穩定誘發的數值錨定偏誤。

#### Figure 1: `boxplots_by_condition.png`
**放置位置**：Section 4.2 Results — 開篇第一張圖

**佐證論點**：
> 錨定偏誤可在 LLMs 上跨多種估計任務被穩定誘發

**Caption 建議**：
> Figure 1. Distribution of percentage estimation errors across four experimental conditions (N=44 questions, 10–30 trials each). The High Anchor condition shows a positive shift in median error relative to Baseline, while the Low Anchor condition shows a negative shift, consistent with anchoring bias. Error is computed as (estimate − true value) / true value × 100%.

**解讀要點**：
- High Anchor（紅色）box 明顯偏右（正向偏移）
- Low Anchor（綠色）box 中位數略低於 Baseline
- 視覺上即可看出 anchor 對估計的拉動方向

---

#### Figure 2: `anchoring_index.png`
**放置位置**：Section 4.2 Results — Figure 1 之後，統計量化段落

**佐證論點**：
> 建立可重現的評估協定：Anchoring Index (AI) 作為標準化量測指標

**Caption 建議**：
> Figure 2. Mean Anchoring Index (AI) by condition for GPT-4o-mini. AI = (median_anchored − median_baseline) / (anchor_value − median_baseline). An AI of 0 indicates no anchoring effect; AI of 1 indicates the model's estimate fully matches the anchor. Error bars denote standard error of the mean. High Anchor (AI=0.41) shows stronger susceptibility than Low Anchor (AI=0.30).

**解讀要點**：
- High AI = 0.41 > Low AI = 0.30 → 高錨比低錨更有效，與人類認知偏誤文獻一致
- 兩者皆顯著大於 0 → 確認錨定效應存在
- 為後續 SGCAP 實驗提供 baseline 參照

---

#### Figure 3: `heatmap_gpt-4o-mini.png`
**放置位置**：Section 4.3 Analysis by Task Domain

**佐證論點**：
> 錨定效應在不同任務領域間的變異性 — 刻畫邊界條件

**Caption 建議**：
> Figure 3. Heatmap of Anchoring Index by event category and anchor direction for GPT-4o-mini. Values closer to 1.0 (darker) indicate stronger anchoring effects. Categories such as temperature forecasting and stock price prediction show consistently high susceptibility (AI ≈ 1.0), while social media metrics and temporal counts show near-zero susceptibility.

**解讀要點**：
- 溫度、股價類：AI ≈ 1.0（完全被錨定）→ 模型對這類任務的不確定性最高
- 社群媒體數據（tweets, views）：AI ≈ 0 → 模型有較強的內部先驗
- 這張圖直接支撐「邊界條件」的貢獻聲明

**重要**：此圖目前以「event_name」為 category，建議重新分組為更抽象的任務類別（temperature, stock price, social media, counting, etc.），論文呈現會更乾淨。

---

### Section 5: Experiment 2 — SGCAP 去偏效果

#### Figure 4: `counter_anchor_effectiveness.png`
**放置位置**：Section 5.2 Results — SGCAP vs High Anchor 比較

**佐證論點**：
> SGCAP 能否達到與外部提供之雙錨點相當的去偏效果

**Caption 建議**：
> Figure 4. Counter-anchor effectiveness: absolute percentage error under High Anchor condition (x-axis) vs. Counter-Anchor condition (y-axis) for each question. Points below the diagonal (red dashed line) indicate questions where the counter-anchor reduced estimation error relative to the high anchor condition. The majority of points cluster near or below the line, suggesting partial debiasing.

**解讀要點**：
- 多數點在對角線附近或以下 → counter-anchor 有效果
- 少數離群點（左上方）表示 counter-anchor 反而引入新偏誤 → 失敗案例
- 這張圖直接回答「何種條件下成功或失敗」

---

#### Figure 5: `scatter_true_vs_estimate.png`
**放置位置**：Section 5.3 Calibration Analysis 或 Appendix

**佐證論點**：
> 整體校準度分析 — 各條件下的系統性偏移

**Caption 建議**：
> Figure 5. True value vs. median estimate across all questions and conditions (log-log scale). The dashed line represents perfect calibration. High Anchor estimates (red) deviate upward from the diagonal more frequently than Baseline (blue), while Counter-Anchor (orange) clusters closer to the perfect prediction line.

**解讀要點**：
- 大部分點沿對角線分布 → 模型整體校準尚可
- High Anchor 的紅色點偏離對角線更遠 → 視覺化錨定拉動
- Counter-Anchor 的橘色點回歸對角線 → SGCAP 的校正效果
- 適合放在 Appendix 作為補充，或放在正文中展示整體校準

---

## 建議追加的圖表（尚未生成）

### Table 1: 主要統計結果摘要表
**放置位置**：Section 4.2 開頭

| Condition | N | MdAPE | Mean APE | AI (mean ± SE) |
|-----------|---|-------|----------|-----------------|
| Baseline | 851 | 9.7% | 54.8% | — |
| High Anchor | 436 | 9.6% | 34.2% | 0.41 ± 0.10 |
| Low Anchor | 435 | 9.7% | 52.5% | 0.30 ± 0.08 |
| Counter-Anchor | 431 | 8.3% | 79.6% | — |

### Table 2: Wilcoxon 統計檢定結果
**放置位置**：Section 4.2 統計檢定段落

| Comparison | W | p-value | Median diff | Significance |
|------------|---|---------|-------------|-------------|
| Baseline vs High | 180.0 | 0.289 | +0.00 | n.s. |
| Baseline vs Low | 68.0 | 0.058 | +0.00 | marginal |
| Baseline vs Counter | 202.5 | 0.065 | +0.09 | marginal |

### Figure 6（建議新增）: Per-question paired comparison
**用途**：展示 61% 題目出現錨定效應、30% 出現雙向錨定
**格式**：Stacked bar 或 pie chart 顯示 ANCHORED / high_only / low_only / none 分布

---

## 目前數據的限制與建議

### ⚠️ 統計顯著性不足
- Wilcoxon p > 0.05（low anchor p=0.058 接近邊界）
- **原因**：多數題目只跑 10 trials，不足以在 per-question median 上產生穩定差異
- **解法**：跑滿 30 trials → `python experiment.py --model gpt-4o-mini --trials 30`

### ⚠️ 只有一個模型
- 論文聲明要測 GPT-4o，目前只有 gpt-4o-mini
- **解法**：`python experiment.py --model gpt-4o --trials 30`

### ⚠️ Heatmap 分類太細
- 目前用 event_name（如 "Apple WWDC"），應改為抽象任務類別
- **建議分類**：
  - Temperature forecasting
  - Stock/financial prediction
  - Social media metrics (views, likes, subscribers)
  - Counting/enumeration
  - Exchange rates
  - Physical measurements

### ⚠️ Counter-Anchor 方法需要改為 SGCAP
- 目前的 counter-anchor 是「給兩個錨 + 警告」（外部雙錨點）
- 論文提出的是 **SGCAP**（自生成反向錨點）→ 需要改 prompt
- SGCAP 三階段：
  1. 讓模型自己生成極端高估計
  2. 讓模型自己生成極端低估計
  3. 以自生成的高低錨點作為參考，產出最終估計

---

## 完整實驗執行清單

```bash
# 1. 跑滿 30 trials（自動跳過已完成的）
python experiment.py --model gpt-4o-mini --trials 30

# 2. 跑 GPT-4o
python experiment.py --model gpt-4o --trials 30

# 3. 實作 SGCAP 條件後重跑
# （修改 experiment.py 中 counter_anchor 的 prompt）
python experiment.py --condition counter_anchor --trials 30

# 4. 重新生成分析
python analysis.py
```
