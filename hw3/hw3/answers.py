r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 128
    hypers["seq_len"] = 128
    hypers["h_dim"] = 256
    hypers["n_layers"] = 2
    hypers["dropout"] = 0.1
    hypers["learn_rate"] = 0.001
    hypers["lr_sched_factor"] = 0.07
    hypers["lr_sched_patience"] = 0.7
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    temperature=0.8
    start_seq="ACT I."
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

We split the corpus into sequences for memory efficiency, as training on the whole text is infeasible.

RNNs/GRUs process data sequentially, learning dependencies in smaller, manageable chunks.

Generalization improves as the model captures localized patterns instead of memorizing the corpus.

Computational feasibility is maintained using truncated backpropagation, making training more effective.

"""

part1_q2 = r"""
**Your answer:**

The generated text can exhibit memory longer than the sequence length because recurrent neural networks maintain a hidden state that carries information across multiple sequences.

Even though training is done on fixed-length sequences, the hidden state persists across time steps, allowing the model to retain context from previous sequences.

GRUs help mitigate vanishing gradients, enabling them to capture long-range dependencies effectively.

This results in text generation that reflects patterns beyond the immediate sequence length.

"""

part1_q3 = r"""
**Your answer:**

We do not shuffle the order of batches when training because the model needs to learn sequential dependencies in the text.  

Shuffling would disrupt the natural order of characters, making it harder for the GRU to capture meaningful transitions between sequences.  

By maintaining the order, the hidden state can effectively carry information from previous batches, allowing the model to learn longer-term dependencies.  

This ensures that training remains consistent with how the text is naturally structured.

"""

part1_q4 = r"""
**Your answer:**

1. Lowering the temperature makes the model sample more confidently from high-probability choices while exploring less. 
This results in more structured and coherent text, reducing randomness. 
As seen in the graph from the Generating text by sampling section, lower temperatures like \(T = 0.3\) create sharper probability distributions.

2. When the temperature is very high, the model samples more randomly, making all characters nearly equally likely. 
This leads to less structured and incoherent text, as seen in the graph from the Generating text by sampling section at \(T = 100\), where the probability distribution is almost uniform. High \(T\) encourages more exploration but sacrifices meaningful structure.

3. When the temperature is very low, the model rarely explores and almost always selects the highest-probability character. 
This results in rigid and repetitive text, as shown in the graph from the Generating text by sampling section, where \(T = 0.3\) forces deterministic selections. Low \(T\) ensures coherence but significantly limits creativity.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = 'https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip'


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 64
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.001
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.8, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The $\sigma^2$ hyperparameter controls the variance in the Gaussian likelihood, affecting reconstruction sharpness. We set $\sigma^2 = 0.001$, which strongly penalizes reconstruction errors, leading to clearer but slightly blurry images due to balancing with the KL-divergence term. 

From the loss formula:

$$
\mathcal{L}(\alpha, \beta) = \mathbb{E}_{\mathbf{x}} \left[ \mathbb{E}_{\mathbf{z} \sim q_{\alpha}} \left[ \frac{1}{2\sigma^2} \|\mathbf{x} - \Psi_{\beta}(\mathbf{z})\|_2^2 \right] + D_{KL}(q_{\alpha}(\mathbf{Z}|\mathbf{x}) \| p(\mathbf{Z})) \right]
$$

a lower $\sigma^2$ enforces precise reconstructions but may overfit, while a higher $\sigma^2$ would result in smoother, less detailed images.

"""

part2_q2 = r"""
**Your answer:**

1. The VAE loss consists of reconstruction loss and KL divergence loss, balancing data fidelity and structured latent space learning.  
   - Reconstruction Loss ensures the decoded output $\mathbf{x}$ is similar to the input by minimizing $\frac{1}{2\sigma^2}\|\mathbf{x} - \Psi_{\beta}(\mathbf{z})\|_2^2$, forcing the model to retain important features.  
   - KL Divergence Loss minimizes $D_{KL}(q_{\alpha}(\mathbf{Z}|\mathbf{x}) \| p(\mathbf{Z}))$, keeping $q_{\alpha}(\mathbf{Z}|\mathbf{x})$ structured to prevent overfitting and ensure smooth latent-space organization.

2. The KL loss term enforces a continuous latent-space distribution, ensuring that $q_{\alpha}(\mathbf{Z}|\mathbf{x})$ remains close to $p(\mathbf{Z})$. This prevents the model from encoding inputs into disjoint regions, maintaining smooth transitions between latent representations. Without it, sampling could become unreliable, leading to unrealistic generations.

3. The benefit of this structured latent space is improved generalization and interpolation, where nearby latent points generate similar outputs. This enables smooth latent-space manipulation, allowing interpolation between different inputs. Without this constraint, the model’s generated outputs could become disorganized and less meaningful.

"""

part2_q3 = r"""
**Your answer:**

We maximize the evidence distribution $p(\mathbf{X})$ because it represents the probability of the observed data, ensuring that the model learns to generate realistic samples. However, directly computing $p(\mathbf{X})$ is intractable, meaning it is computationally infeasible due to the need to integrate over all possible latent variables $\mathbf{Z}$. To address this, we use the ELBO (Evidence Lower Bound) as an optimization target:

$$
\log p(\mathbf{X}) \geq \mathbb{E}_{\mathbf{z} \sim q_{\alpha}}\left[\log p_{\beta}(\mathbf{X} \mid \mathbf{z})\right] - D_{KL}\left(q_{\alpha}(\mathbf{Z} \mid \mathbf{X}) \,\Vert\, p(\mathbf{Z})\right)
$$

By maximizing the ELBO, we effectively maximize $p(\mathbf{X})$, ensuring that the model optimizes both reconstruction quality and a well-structured latent space for meaningful generation.

"""

part2_q4 = r"""
**Your answer:**

We model $\log \sigma_{\alpha}^2(x)$ instead of directly modeling $\sigma_{\alpha}^2(x)$ for three key reasons:

1. **Numerical Stability:** Ensures that the variance remains strictly positive, preventing instability in gradient updates and avoiding invalid negative values.
2. **Unconstrained Optimization:** Since $\log (\sigma^2)$ can take any real value, it allows smooth optimization in $\mathbb{R}$, avoiding the need to enforce positivity constraints.
3. **Efficient KL Divergence Computation:** The KL divergence formula used in VAEs requires log-variance for closed-form computation, making backpropagation more stable and efficient.

"""

# Part 3 answers
def part3_gan_hyperparams():
    # hypers = dict(
    #     batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    # )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=32,
        h_dim=512,
        z_dim=64,
        x_sigma2=0.0001,
        learn_rate=0.0001,
        betas=(0.9, 0.999),
        data_label=1,
        label_noise=0.1,
        discriminator_optimizer={
            'type': 'Adam',
            'lr': 0.0001,
            'betas': (0.5, 0.999)
        },
        generator_optimizer={
            'type': 'Adam',
            'lr': 0.0005,
            'betas': (0.5, 0.999)
        }
    )
    # ========================
    return hypers

part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""

part3_q3 = r"""
**Your answer:**



"""



PART3_CUSTOM_DATA_URL = 'https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip'


def part4_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers["embed_dim"] = 512
    hypers["num_heads"] = 8
    hypers["num_layers"] = 4
    hypers["hidden_dim"] = 128
    hypers["window_size"] = 128
    hypers["droupout"] = 0.1
    hypers["lr"] = 0.0001

    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

1. When do we need to maintain gradients, and why?  
Gradients are maintained when training the generator, allowing backpropagation to update its parameters and improve its ability to fool the discriminator. This is essential for minimizing the generator’s loss during adversarial training.

2. When are gradients discarded, and why?  
Gradients are discarded during discriminator training and inference because the generator is not being updated. This reduces computational cost and avoids unnecessary gradient accumulation.

"""

part3_q2 = r"""
**Your answer:**

1. When training a GAN to generate images, a possible reason we should not stop training solely based on the generator loss reaching a threshold is that generator loss does not directly reflect image quality. This is probably because the loss can decrease even when the generator produces repetitive or low-quality samples due to mode collapse. We can assume that stopping training prematurely might miss the opportunity for the generator to improve diversity and realism. Thus, it's more reliable to assess image quality visually alongside monitoring loss trends.

2. If the discriminator loss remains at a constant value while the generator loss decreases, a possible explanation is that the discriminator has reached a stable state with high accuracy. This is probably because it confidently classifies real and fake samples, keeping its loss low and steady. We can assume that the generator is gradually improving to match the discriminator's performance, as shown by its decreasing loss. This pattern indicates that the generator is still learning and refining its outputs despite the discriminator’s consistent confidence.

"""

part3_q3 = r"""
**Your answer:**

The main difference is that VAE outputs are smoother but less detailed, while GAN outputs are sharper and more realistic. This is because VAEs minimize reconstruction loss, which smooths fine details, while GANs rely on adversarial training to create high-quality samples. The adversarial loss in GANs encourages diversity, but they can be unstable, unlike VAEs that provide consistent results.

"""


part4_q1 = r"""
**Your answer:**

Stacking encoder layers with **sliding-window attention** results in a broader context because each layer combines information from neighboring windows. In the first layer, each token attends only to a small local context, such as nearby tokens within a limited range. In the second layer, overlapping windows allow information from distant tokens to merge. Consequently, stacking multiple layers increases the effective receptive field, enabling the final layer to capture a much larger context.

"""

part4_q2 = r"""
**Your answer:**

The scaled dot product attention in the sliding-window approach is computed only within a fixed-size window around each token, reducing the computational complexity from $O(n^2)$ to $O(nw)$, where $n$ is the sequence length and $w$ is the window size.

#### Let's start off with the original sliding-window approach  
For each token $i$, the attention is restricted to the window:
$$
\mathcal{N}(i) = \{ j \mid |i - j| \leq \frac{w}{2} \}
$$
The normalized dot product is computed as:
$$
b(q, k, w) =
\begin{cases}
\displaystyle \frac{q \cdot k^T}{\sqrt{d_k}} & \text{if } d(q, k) \leq \frac{w}{2}, \\
-\infty & \text{otherwise}.
\end{cases}
$$
This ensures that each token attends only to its local neighbors, without any global context.

---

### Proposed Hybrid Variation: Sliding-Window + Sparse Global Attention  
To include a more global context while maintaining $O(nw)$ complexity, we propose adding global attention to a subset of tokens alongside the sliding-window structure.

#### Key Modifications:
1. **Global Token Set $\mathcal{G}$**  
   A small set of predefined positions will have global attention, attending to all other tokens and being attended by them.

2. **Local + Global Attention Coverage**  
   The attention pattern for each token $i$ becomes:
   $$
   \mathcal{C}(i) = \mathcal{N}(i) \cup \mathcal{G}
   $$
   - Global tokens attend to the entire sequence.
   - Non-global tokens attend to their local window and global tokens.

3. **Computational Complexity**  
   Since $\mathcal{G}$ is small, the added cost remains linear. The overall complexity is still $O(nw)$, preserving efficiency while improving context.
   
"""

part5_q1= r"""
**Your answer:**

The fine-tuned encoder performed better than the encoder trained from scratch, showing faster convergence and higher accuracy. This is because the fine-tuned model benefits from pre-trained knowledge, which captures general language patterns and representations. This prior knowledge allows the model to adapt quickly to similar tasks with minimal training effort. However, this phenomenon depends on the task. Fine-tuning is most effective when the downstream task aligns with the pre-training data. For highly domain-specific tasks, training from scratch may sometimes yield better results if pre-trained knowledge is less relevant or even detrimental.

"""

part5_q2= r"""
**Your answer:**

Assuming the model fine-tunes successfully, freezing the final layers and fine-tuning internal layers like the multi-headed attention block might limit task-specific adaptation. Since the final layers capture high-level features relevant to the task, preventing their adjustment could reduce the model's ability to fully adapt. Additionally, modifying deeper internal layers increases the risk of overfitting to non-relevant patterns, especially with limited task-specific data, making this approach potentially less optimal.

"""

part5_q3= r"""
**Your answer:**

No, BERT alone cannot handle machine translation because it is not designed for autoregressive generation. It predicts masked tokens within a sequence but cannot generate target sequences step by step. In my opinion, a possible solution would be to modify BERT into an encoder-decoder structure by adding a decoder block for sequential prediction. Additionally, pre-training should include sequence-to-sequence tasks to improve generation performance.

"""

part5_q4 = r"""
**Your answer:**

A main reason to choose RNN over a Transformer is its inherent ability to model continuous and time-dependent data, such as real-time signal processing or continuous speech generation, where sequential order is crucial. RNNs maintain a persistent internal state, which allows them to naturally handle temporal patterns without relying on position encoding.

"""

part5_q5 = r"""
**Your answer:**

**Next Sentence Prediction (NSP)** is a pre-training task in BERT where the model predicts whether a sentence \(B\) logically follows sentence \(A\). It’s framed as a binary classification problem with *IsNext* or *NotNext* labels, using **binary cross-entropy loss** for optimization.

**Is it crucial?**  
In my opinion, it depends on the task. NSP is valuable for tasks requiring sentence-level understanding, such as **question answering**. However, for token-level tasks like **named entity recognition**, NSP contributes less since relationships between sentences are less critical.

"""


# ==============
