# Build Your Own AI Agent From Scratch — Reading Assignments

Each assignment targets the 2-5 pages that contain the core insight. Read those pages deeply rather than skimming an entire paper.

---

### BUILD-01: Make It Speak
**Required Reading:** "Language Models are Unsupervised Multitask Learners" (GPT-2) — Section 2: Approach (pages 3-4)
**Why this section:** This is where language modeling clicks — the equation p(x) = ∏ p(sₙ|s₁,...,sₙ₋₁) and the insight that all tasks can be framed as next-token prediction.
**Key passage:**
> "Language modeling is usually framed as unsupervised distribution estimation from a set of examples (x₁, x₂, ..., xₙ) each composed of variable length sequences of symbols (s₁, s₂, ..., sₙ). Since language has a natural sequential ordering, it is common to factorize the joint probabilities over symbols as the product of conditional probabilities... Learning to perform a single task can be expressed in a probabilistic framework as estimating a conditional distribution p(output|input). Since a general system should be able to perform many different tasks, even for the same input, it should condition not only on the input but also on the task to be performed. That is, it should model p(output|input, task)."

**Also recommended:** The Illustrated Transformer by Jay Alammar (offline mirror: `papers/llm-essentials/illustrated-transformer/`) — a visual walkthrough that makes the architecture intuitive before you read equations.

**Optional deep dive:** GPT-2 Section 3 (results) to see zero-shot performance scaling with model size.

---

### BUILD-02: Make It Smarter
**Required Reading:** "Attention Is All You Need" — Section 3.2: Attention (pages 3-5)
**Why this section:** This is THE equation of modern AI — Scaled Dot-Product Attention and Multi-Head Attention are the entire mechanism by which transformers think.
**Key passage:**
> "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key... We call our particular attention 'Scaled Dot-Product Attention'. The input consists of queries and keys of dimension dₖ, and values of dimension dᵥ. We compute the dot products of the query with all keys, divide each by √dₖ, and apply a softmax function to obtain the weights on the values:
>
> Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V
>
> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."

**Also recommended:** The Illustrated Transformer blog (offline mirror) — the visual walkthrough of self-attention with colored diagrams.

**Optional deep dive:** RoFormer — Section 3.2: Rotary Position Embedding (pages 4-5). RoPE encodes position by *rotating* query/key vectors rather than adding position embeddings. The key insight: fq(xₘ, m) = (Wq·xₘ)·e^(imθ), which naturally encodes relative position m−n in the dot product. This is how every modern LLM handles position.

---

### BUILD-03: Give It a Window
**Required Reading:** "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" — Section 1 (pages 1-2) + Section 3.1-3.2 (pages 4-6)
**Why this section:** The insight that attention is memory-bound, not compute-bound, changed everything about how we build long-context models.
**Key passage:**
> "We argue that a missing principle is making attention algorithms IO-aware—that is, carefully accounting for reads and writes to different levels of fast and slow memory (e.g., between fast GPU on-chip SRAM and relatively slow GPU high bandwidth memory, or HBM). On modern GPUs, compute speed has out-paced memory speed, and most operations in Transformers are bottlenecked by memory accesses. IO-aware algorithms have been critical for similar memory-bound operations... We propose FlashAttention, a new attention algorithm that computes exact attention with far fewer memory accesses. Our main goal is to avoid reading and writing the attention matrix to and from HBM."

And the theoretical payoff (Section 3.2):
> "Standard attention requires Θ(Nd + N²) HBM accesses, while FlashAttention requires Θ(N²d²M⁻¹) HBM accesses. For typical values of d (64-128) and M (around 100KB), d² is many times smaller than M, and thus FlashAttention requires many times fewer HBM accesses than standard implementation."

**Optional deep dive:** FlashAttention-2 paper for the improved algorithm; any discussion of Ring Attention for multi-GPU long context.

---

### BUILD-04: Let It Talk to You
**Required Reading:** No paper — this is systems engineering (KV-cache, streaming, tokenization).
**Context from InstructGPT** (Section 1, page 1): The gap between language modeling objective and "follow the user's instructions helpfully and safely" is what makes serving an interactive system different from training one.

**Optional deep dive:** The vLLM paper on PagedAttention for efficient serving; HuggingFace's text-generation-inference documentation.

---

### BUILD-05: Let It Act
**Required Reading:** "ReAct: Synergizing Reasoning and Acting in Language Models" — Sections 1-2 (pages 1-4)
**Why this section:** This paper introduces the Thought→Action→Observation loop that became the standard agent pattern. Section 2 is one page and defines the entire framework.
**Key passage:**
> "The idea of ReAct is simple: we augment the agent's action space to  = A ∪ L, where L is the space of language. An action âₜ ∈ L in the language space, which we will refer to as a thought or a reasoning trace, does not affect the external environment, thus leading to no observation feedback. Instead, a thought âₜ aims to compose useful information by reasoning over the current context cₜ, and update the context cₜ₊₁ = (cₜ, âₜ) to support future reasoning or acting."

**Also required:** "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" — Section 1-2 (pages 1-3)
**Why:** The discovery that simply showing a model step-by-step reasoning in the prompt unlocks reasoning abilities. Figure 1 tells the whole story.
**Key passage:**
> "We explore how generating a chain of thought—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain-of-thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting."

**Also required:** "Toolformer: Language Models Can Teach Themselves to Use Tools" — Section 1-2 (pages 1-3)
**Why:** The self-supervised approach to tool use — the model learns *when* to call APIs by checking if the API result reduces its next-token prediction loss.
**Key passage:**
> "We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API."

**Optional deep dive:** The Toolformer filtering mechanism (Section 2, Figure 2) — how they decide which API calls actually help.

---

### BUILD-06: Let It Remember
**Required Reading:** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG) — Sections 1-2 (pages 1-4)
**Why this section:** The architecture diagram (Figure 1) and the two RAG formulations (RAG-Sequence vs RAG-Token) define how to give a model external memory.
**Key passage:**
> "Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remain open research problems... We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-trained parametric and non-parametric memory for language generation."

The key architectural insight (Section 2.1):
> "pRAG-Sequence(y|x) ≈ Σ pη(z|x) pθ(y|x,z) — the top K documents are retrieved using the retriever, and the generator produces the output sequence probability for each document, which are then marginalized."

**Optional deep dive:** DPR (Dense Passage Retrieval) paper for the retriever side; REALM for the idea of pre-training with retrieval.

---

### BUILD-07: Let It See
**Required Reading:** "The Platonic Representation Hypothesis" — Sections 1-2 (pages 1-4)
**Why this section:** The convergence thesis — that different neural networks trained on different modalities are converging toward the same representation of reality — is the theoretical foundation for multimodal AI.
**Key passage:**
> "We argue that representations in AI models, particularly deep networks, are converging. First, we survey many examples of convergence in the literature: over time and across multiple domains, the ways by which different neural networks represent data are becoming more aligned. Next, we demonstrate convergence across data modalities: as vision models and language models get larger, they measure distance between datapoints in a more and more alike way. We hypothesize that this convergence is driving toward a shared statistical model of reality, akin to Plato's concept of an ideal reality."

**Also recommended:** "Scaling Monosemanticity" (offline mirror: `papers/llm-essentials/scaling-monosemanticity/`) — the Golden Gate Bridge feature section.
**Why:** They found a single feature in Claude 3 Sonnet that activates for the Golden Gate Bridge across *every language* — Chinese, Japanese, Korean, Russian, Vietnamese, Greek — demonstrating that features inside large models are multilingual and multimodal. When they amplified this feature, Claude started claiming to *be* the Golden Gate Bridge. This is the most vivid demonstration of how models represent concepts internally.

**Optional deep dive:** Scaling Monosemanticity's full "Safety-Relevant Features" section — features for deception, sycophancy, bias, and dangerous content found inside Claude.

---

### BUILD-08: Make It Scale
**Required Reading #1:** "Scaling Laws for Neural Language Models" — Section 1: Introduction + Summary (pages 2-5)
**Why this section:** The discovery that loss follows clean power laws across 7 orders of magnitude — and the practical implications for how to spend compute.
**Key passage:**
> "Performance depends strongly on scale, weakly on model shape: Model performance depends most strongly on scale, which consists of three factors: the number of model parameters N (excluding embeddings), the size of the dataset D, and the amount of compute C used for training. Within reasonable limits, performance depends very weakly on other architectural hyperparameters such as depth vs. width."

And the compute-efficiency finding:
> "Convergence is inefficient: When working within a fixed compute budget C but without any other restrictions on the model size N or available data D, we attain optimal performance by training very large models and stopping significantly short of convergence. Maximally compute-efficient training would therefore be far more sample efficient than one might expect based on training small models to convergence, with data requirements growing very slowly as D ∼ C^0.27 with training compute."

**Required Reading #2:** "Training Compute-Optimal Large Language Models" (Chinchilla) — Sections 1 + 3.1 (pages 1-2, 5-6)
**Why this section:** Chinchilla *corrected* the Scaling Laws paper — model size and data should scale equally, not asymmetrically. This single insight reshaped the entire field.
**Key passage:**
> "We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled."

**Read these two papers together.** Kaplan et al. (2020) discovered the power laws. Hoffmann et al. (2022) corrected the allocation recipe. Together they tell you: *loss is predictable, and the optimal way to spend compute is to scale model and data in lockstep.*

**Optional deep dive:** Chinchilla Table 1 — comparing model sizes and training tokens across GPT-3, Gopher, Jurassic, etc. to see how undertrained they were.

---

### BUILD-09: Make It Safe
**Required Reading #1:** "Training language models to follow instructions with human feedback" (InstructGPT) — Section 1 (pages 1-4) + Section 3.5: Models (pages 8-9)
**Why this section:** The 3-step RLHF pipeline (SFT → Reward Model → PPO) and the stunning result that a 1.3B model trained with human feedback beats a 175B model without it.
**Key passage:**
> "Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users... In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters."

And the method (Section 3.5):
> "Reinforcement learning (RL). We fine-tuned the SFT model on our environment using PPO. The environment is a bandit environment which presents a random customer prompt and expects a response to the prompt. Given the prompt and response, it produces a reward determined by the reward model and ends the episode. In addition, we add a per-token KL penalty from the SFT model at each token to mitigate over-optimization of the reward model."

**Required Reading #2:** "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (DPO) — Section 1 + Section 4 (pages 1-2, 4-5)
**Why this section:** DPO eliminates the entire RL loop by showing the reward model is implicit in the policy. The key derivation fits on one page.
**Key passage:**
> "Our key insight is to leverage an analytical mapping from reward functions to optimal policies, which enables us to transform a loss function over reward functions into a loss function over policies. This change-of-variables approach avoids fitting an explicit, standalone reward model, while still optimizing under existing models of human preferences... In essence, the policy network represents both the language model and the (implicit) reward."

The DPO loss itself:
> L_DPO(πθ; πref) = −E[log σ(β log(πθ(yw|x)/πref(yw|x)) − β log(πθ(yl|x)/πref(yl|x)))]

**Also recommended:** Scaling Monosemanticity — "Safety-Relevant Features" section (see offline mirror). They found features for deception, power-seeking, sycophancy, and dangerous content inside Claude 3 Sonnet, and demonstrated that features can be used to *detect and correct* deceptive behavior. Key quote from the page:
> "Some of the features we find are of particular interest because they may be safety-relevant – that is, they are plausibly connected to a range of ways in which modern AI systems may cause harm. In particular, we find features related to security vulnerabilities and backdoors in code; bias; lying, deception, and power-seeking (including treacherous turns); sycophancy; and dangerous / criminal content."

**Optional deep dive:** Constitutional AI (Anthropic) for the RLAIF approach; the full DPO derivation in Section 4 for the beautiful change-of-variables trick.

---

### BUILD-10: Set It Free
**Required Reading:** No specific paper section — this module is about deployment, monitoring, and autonomous operation.
**Relevant context from ReAct** (Section 2): The ReAct framework's distinction between actions that affect the environment and thoughts that don't is directly relevant to building safe autonomous agents — you want the agent to *think* freely but *act* cautiously.
**Relevant context from InstructGPT** (Section 5.3): The discussion of limitations — InstructGPT still makes simple mistakes, hallucinates, and can be brittle to prompt phrasing — is essential reading for anyone deploying an agent in production.

**Optional deep dive:** The Voyager paper (autonomous Minecraft agent) for long-horizon autonomous operation; the AutoGPT/BabyAGI ecosystem for practical patterns in autonomous agent loops.
