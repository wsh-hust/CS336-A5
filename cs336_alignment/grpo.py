from __future__ import annotations

import torch
import math
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from typing import Callable, Literal


"""
强化学习，目的是学习一个策略，使得目标函数也就是奖励函数对应的期望值尽可能大。而期望值的计算是：J(θ)=Eτ∼πθ[R(τ)]，其中τ是一个轨迹，R(τ)是轨迹的奖励函数，πθ是策略函数。我们希望最大化J(θ)，也就是最大化期望奖励。
实际中要找到这个最好的策略函数，需要通过j对参数θ进行求导计算梯度，然后使用梯度上升法来更新参数θ。这个梯度的计算是通过策略梯度定理来实现的，公式为：
∇θJ(θ)=Eτ∼πθ[∇θlogπθ(τ)R(τ)]
也就是对轨迹的奖励函数R(τ)

对于语言模型来说，这里的轨迹τ是一个序列，每个位置是一个token，也就是模型输出的完整回答(由一个个token组成)，而奖励函数R(τ)是一个标量，表示这个回答的好坏。我们希望最大化这个奖励函数的期望值，也就是希望模型输出的回答尽可能好.

具体优化策略，也就是优化语言模型的参数θ的过程是：
1. 采样：从当前策略πθ中采样出一批轨迹，每个轨迹是一个token序列。
2. 计算奖励：对每个轨迹，根据奖励函数R(τ)计算出轨迹的奖励。
3. 计算梯度：根据策略梯度定理，计算出每个轨迹的梯度∇θlogπθ(τ)R(τ)。
4. 更新参数：使用梯度上升法，根据计算出的梯度更新模型的

然后这里又分为同策略和异策略两种情况：
1. 同策略：采样的轨迹是从当前策略πθ中采样
2. 异策略：采样的轨迹是从其他策略πθ中采样
对于异策略，因为从初始策略x中采用得到一批数据，然后用其中一批数据更新策略后，就得到了新的策略y，而y是与x不同的策略。
如果是同策略，则继续从y中采样数据，更新策略为z。这样很慢，因为每一次采样一批数据，都是输入一个prompt，得到对应的回答，而生成回答是一个token一个token的输出的，所以很慢

所以采用异策略，也就是对于刚刚得到的y，依然用从初始策略x中采样得到的数据进行更新。但是由于x和y不完全一样，对于初始数据(从x采样得到的轨迹)的概率不一样，所以还要乘一个概率权重。

对于每一次更新，一个关键步骤就是计算一条轨迹中各个token的log p，具体实现方式为：
首先对于采样的数据，其构成是这样的：
对于一个prompt，采样模型生成的回答，可能有很多个，分别记录下来
每一次更新时，选择一批prompt和对应的回答，一个prompt和一个对应回答加起来就是一个轨迹，而更新是通过梯度上升，要得到梯度就需要得到这个轨迹中每个token的log p(不包括prompt中的token)，实现方式就是先把prompt和对应的回答分别tokenize，然后再拼接起来送入语言模型，根据实验1可以知道，对于一个轨迹(一个prompt和一个对应回答)，假设其长度为L，语言模型会输出一个(L,V)的logits矩阵，其中V是词表大小，然后对这个logits矩阵进行softmax归一化得到概率P，然后再log处理，得到每个位置的log p，因为完整的回答已经知道了，第i行的回答就是第i+1个token，所以就可以根据回答中每个token的id去gather对应位置的log p，就得到了这个轨迹中每个token的log p。
也就是p(yt|y1,...,yt-1,x)就是输入的(一个prompt和一个对应回答)得到的输出矩阵的第len(prompt)+t-1行的第yt列的值，也就是log p(yt|y1,...,yt-1,x)。

"""
"""
tokenize_prompt_and_output 函数的作用：
把 prompt 和 output 分别进行 tokenize，然后拼接起来，得到一个完整的轨迹的 token id 矩阵，并且对其进行 padding，使得同一个batch中每个轨迹的长度一致。然后根据这个 token id 矩阵，生成 labels 矩阵和 response_mask 矩阵。labels 矩阵是去掉第一个 token 的 token id 矩阵，response_mask 矩阵是一个布尔矩阵，表示哪些位置是 output 的 token。
得到的input_ids, labels, response_mask 矩阵的维度都是 (B,L)，其中 B 是 batch size，L 是最长的轨迹长度。
input_ids 矩阵是模型的输入，也就是一个prompt和一个对应回答的token id矩阵。把他输入语言模型，得到的logits矩阵的维度是(B,L,V)，其中V是词表大小。第i行就是前i个token对第i+1
个的预测得分；
labels： 每个位置“应该预测”的 token id，其实就是这个回答(初始采样得到的数据，用来异策略更新策略)的每一个token的下一个token的id。第i个token对应的label就是第i+1个token的id。也就是labels[i] = input_ids[i+1]
response_mask： 掩码，因为不需要对prompt中的token计算loss，因为模型不需要学习提示词。但是要从promt的最后一个token开始计算，因为prompt的最后一个token是预测回答的第一个token。也就是P(y0|x)

假设x=prompt+ans(即提示词的token id组和对应的语言模型的回答的token id组)
则inputs=x[:-1]，因为最后一个token不需要预测，所以去掉最后一个token
labels=x[1:]，因为第一个token预测的是下一个token，所以去掉第一个token刚好对齐
response_mask=[False]*len(prompt)+[True]*len(ans)
response_mask=response_mask[1:]，这里虽然提示词长度为n，但是要从提示词的最后一个token开始计算，因为他预测的就是回答的第一个token。也就是P(y0|x)，是要计算loss的
"""
def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    assert len(prompt_strs) == len(output_strs)
    prompt_ids=[]
    output_ids=[]
    # 通过encode把prompt和output分别tokenize，得到对应的token id列表
    for pro in prompt_strs:
        prompt_ids.append(tokenizer.encode(pro, add_special_tokens=False))
    for opt in output_strs:
        output_ids.append(tokenizer.encode(opt, add_special_tokens=False))
    input_ids=[]
    # 拼接prompt和output的token id列表，得到一个完整的轨迹的token id列表
    for prompt, output in zip(prompt_ids, output_ids):
        input_ids.append(prompt + output)
    maxlen=0
    # 找到最长的轨迹长度，因为一个batch中每个轨迹的长度可能不一样，所以要进行padding，使得每个轨迹的长度一致
    for ids in input_ids:
        maxlen=max(maxlen,len(ids))
    pad_id = tokenizer.pad_token_id
    for ids in input_ids:
        if len(ids)<maxlen:
            ids+=[pad_id]*(maxlen-len(ids))
    # 计算标签，也就是实际生成的回答中每个token的下一个token，后面找log p的时候用到
    labels=[]
    for ids in input_ids:
        labels.append(ids[1:])
    # 计算掩码，要从prompt的最后一个token开始计算
    response_mask=[]
    for prompt, output in zip(prompt_ids, output_ids):
        mask=[False]*len(prompt)+[True]*len(output)
        mask += [False]*(maxlen-len(mask))
        response_mask.append(mask[1:])

    input_ids=[ids[:-1] for ids in input_ids]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
    }


"""
get_response_log_probs 函数的作用：
输入一个语言模型，一个token id矩阵input_ids和一个标签矩阵labels，计算每个位置的log p以及对应token的熵。
计算梯度时需要用到每个位置的log p，这里的p是已经有了提示词prompt以及生成的前t-1个回答token时，第t个回答token是采样得到的轨迹中的那个token(也就是label中t-1位置对应的那个token id)的概率
具体实现方式是将inputs送入语言模型，得到logits矩阵，然后对logits矩阵进行softmax归一化得到概率P，再log处理，得到每个位置的log p。然后根据labels矩阵中每个位置的token id去gather对应位置的log p，就得到了这个轨迹中每个token的log p。
而token的熵的计算公式是-P log P，分两部分看，P和log（1/p）分别有什么含义？
"""
"""
P(v)：模型此刻对“下一个 token 是 v”的信念，只起加权作用——出现越频繁的候选，对平均不确定性影响越大。
log(1/P(v)) = −log P(v)：候选 v 的自信息 / 惊异度（surprisal）——如果真采到了 v，这个事件有多意外、携带多少信息。P=1 → 0（必然事件零信息）；P=0.01 → ≈4.6 nats。它就是监督训练里 NLL 损失那一项的形式。
熵 = 按概率加权的平均惊异度：模型在这个位置预测下一个 token 的平均不确定性。分布越尖熵越小（one-hot → 0），越平越大（均匀分布 → log V）。
一句话：P 回答“多可能出现”，log(1/P) 回答“真出现了有多意外”，熵是后者的期望。例：分布 (0.5, 0.25, 0.25) 的熵 = 0.5·0.69 + 0.25·1.39 × 2 ≈ 1.04 nats，比“接近确定”大、比“完全均匀”（log 3 ≈ 1.10）略小。

为什么作业要记录它
熵不进 loss，是训练健康度的诊断指标：

熵持续快速下跌 → entropy collapse：策略过早确定化、停止探索，容易卡在次优策略或走向 reward hacking；
熵异常上升：策略在“变随机”，往往是 clip/KL 约束没起作用。
"""
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits  # (B,L,V)
    # 取概率，要对logits进行softmax，防止溢出，先减最大值
    # logits -= logits.max(dim=-1, keepdim=True)[0]
    # 然后对logits进行softmax
    # log_softmax 内部已做 max-shift,数值稳定
    log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)  # (B,L,V)
 
    # 取出每个 label 对应 token 的 log p
    token_log_probs = log_probs_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B,L)
 
    out = {"log_probs": token_log_probs}
    # token的熵是对词表里所有词（比如 32,000 个词）的 - p log p 进行求和
    # 知乎作业五解释
    # 怎么计算这个熵？有什么用？
    if return_token_entropy:
        probs = log_probs_all.exp()
        out["token_entropy"] = -(probs * log_probs_all).sum(dim=-1)  # (B,L)
    return out
"""
维度变化：以 B=2, L=4, V=5 为例：
input_ids              (2, 4)         token id 矩阵
  │ model(input_ids).logits
logits                 (2, 4, 5)      每个位置对 5 个词各打一个分
  │ log_softmax(dim=-1)               沿词表维归一化
log_probs_all          (2, 4, 5)      每个位置的log p,每行5个值加起来= log 1 = 0
  │
labels                 (2, 4)         每个位置“应该预测”的 token id
  │ unsqueeze(-1)
labels.unsqueeze(-1)   (2, 4, 1)      把每个标量包成只有 1 个元素的下标
  │ gather(-1, index)
gathered               (2, 4, 1)      每个位置取到 1 个 log p
  │ squeeze(-1)
token_log_probs        (2, 4)         每行每列一个 log p

"""


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],#模型采样出来的一批回答
    repeated_ground_truths: list[str],#正确答案，如72
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    rollout = 从当前策略采样的、一条完整的模型回答。
    计算每个轨迹的奖励
    reward_fn: 打分器，签名是“两个字符串进、一个字典出”。调用方式固定为：
    reward_fn(response, ground_truth)
    两个参数的含义:
    第一个 response：模型原始输出的完整字符串。对 r1_zero 就是 <think>…</think><answer>…</answer> 这一整段(实际采样时 vLLM 的 stop 串 </answer> 会包含在文本里)；对 question_only 是包含 \boxed{} 的回答。
    第二个 ground_truth：只有答案本体的字符串，如 "72"、"0.5"。注意不是 GSM8K 原始 answer 字段(那个是 "推理过程 #### 72")，调用前必须先按 #### 切开、取最后一段、strip。
    返回值字典固定三个键：
    "reward"        总奖励(本作业 = answer_reward,无部分分)
    "format_reward" 格式奖励:是否满足提示要求的标签/boxed 结构
    "answer_reward" 答案奖励:解析出的答案是否与 ground truth 等价

    rollout_responses: 策略采样出来的一批回答文本.长度定义为：
    rollout_batch_size = n_prompts_per_rollout_batch × group_size

    repeated_ground_truths: 与上者等长、逐位对应的标准答案列表。之所以叫 “repeated”：每个 prompt 的答案在列表里连续重复了 group_size 次(因为同一个 prompt 采样了 G 条回答，但答案只有一个)

    compute_rollout_rewards 返回一个二元组：
    return raw_rewards, metadata
    1. raw_rewards: torch.Tensor，形状 (rollout_batch_size,)，dtype float32
    每条 rollout 一个未归一化的总奖励，顺序与 rollout_responses 完全一致(prompt-major)。
    raw_rewards[i] = reward_fn(rollout_responses[i], repeated_ground_truths[i])["reward"]
    对真实奖励函数取值是 0/1

    2. metadata: dict[str, float]，标量统计，只用于日志
    至少包含两项：
    {
        "mean_reward": 总奖励均值,
        "mean_format_reward": 格式奖励均值,
    }
    可以再加 mean_answer_reward、max/min、正确率等，键名自定(测试不校验)。值用 Python float(如 float(raw_rewards.mean()))。

    注意两个不返回的东西：不做基线/归一化(那是下一步 compute_group_normalized_rewards 的职责)，也不返回逐条的 format/answer 奖励张量——它们的逐条明细只在本函数内部用于算 metadata。

    函数签名里根本没有 group_size，所以它拿不到“哪几条属于同一个 prompt”，自然算不了组内统计。它只负责“整批打分的汇总”，用于日志里报“训练奖励(total/format)”。
    """
    # reward_fn 打分器返回是一个字典，至少包含三个键：
    # "reward"        总奖励(本作业 = answer_reward,无部分分)
    # "format_reward" 格式奖励:是否满足提示要求的标签/boxed 结构
    # "answer_reward" 答案奖励:解析出的答案是否与 ground truth 等价
    rewards,format_rewards,answer_rewards = [],[],[]
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        # response 是模型采样出来的一条回答，ground_truth 是正确答案，如72、0.5
        reward_dict = reward_fn(response, ground_truth)
        # 分别得到这一个回答的总奖励、格式奖励和答案奖励，并加入对应的列表
        rewards.append(reward_dict["reward"])
        format_rewards.append(reward_dict["format_reward"])
        answer_rewards.append(reward_dict["answer_reward"])

    raw_rewards = torch.tensor(rewards, dtype=torch.float32)
    metadata = {
        # 这一批回答的总奖励均值、格式奖励均值和答案奖励均值
        # 注意是采集的这一批prompt的所有回答的均值，不是group组内的均值
        "mean_reward": float(raw_rewards.mean()),
        "mean_format_reward": float(torch.tensor(format_rewards, dtype=torch.float32).mean()),
        "mean_answer_reward": float(torch.tensor(answer_rewards, dtype=torch.float32).mean()),
    }

    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int, #Literal 是 typing 里的字面量类型标注，用来声明“这个参数只能取某几个具体的值”。
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):
    """
    把展平的奖励 `(B*G,)` 整形回 `(B, G)` 组，在组内施加基线和归一化，输出优势。

    group_size: 每个 prompt 采样的回答数量。每个 rollout 包含 group_size 个回答。

    baseline: 组内基线的计算方式。可选：
        "mean" - 组内均值
        "none" - 不做基线
        A_ij = R[i,j] - mu_i
        A_ij = R[i,j] - 0

    advantage_normalizer: 组内优势的归一化方式。可选：
        "std"  - 组内标准差归一化
        "mean" - 组内均值归一化
        "none" - 不做归一化
        A_ij = (R[i,j] - mu_i) / (std_i + eps)       # std 模式
        A_ij = (R[i,j] - mu_i) / (mu_i + eps)        # mean 模式

    advantage_eps: 避免除零的 epsilon。归一化时 denominator = std/mean + advantage_eps

    return advantages, metadata
    advantages: (rollout_batch_size,),与 raw_rewards 同序、同形状、同 dtype/device。它是后面 compute_policy_gradient_loss 的输入：
    per_token_loss[b, t] = -A_b * log_prob[b, t]

    metadata: dict[str, float],日志用，键名自定。建议记录：
    mean_advantage、std_advantage、min/max;
    frac_zero_advantage(优势为 0 的组比例 / 序列比例)——这是判断“模型是不是全对/全错”的重要信号;
    mean_group_std、frac_groups_all_correct、frac_groups_all_wrong。
    """
    # 因为是要分组进行优势计算(减去的均值和标准差都是组内的原始奖励计算得到的)，所以要先将奖励整形为 (B, G) 组，此时相当于每一行是一个 prompt，该行的每一列是这个 prompt 的一个采样回答的奖励值
    rewards = raw_rewards.view(-1, group_size)  # (B*G,)->(B,G)

    # 计算baseline，也就是组内的均值
    if baseline == "mean":
        mu = rewards.mean(dim=1, keepdim=True)  # (B,1)
    elif baseline == "none":
        mu = torch.zeros_like(rewards) 
    else:
        raise NotImplementedError(f"Unsupported baseline: {baseline}")

    # 计算标准差
    if advantage_normalizer == "std":
        std = rewards.std(dim=-1, keepdim=True)  # (B,1)
        denom = std + advantage_eps
    elif advantage_normalizer == "mean":
        std = rewards.mean(dim=-1, keepdim=True)  # (B,1)
        denom = std + advantage_eps
    elif advantage_normalizer == "none":
        denom = None
    else:
        raise NotImplementedError(f"Unsupported advantage_normalizer: {advantage_normalizer}")

    # 计算优势，也就是每个回答的奖励减去组内均值，再除以组内标准差
    if denom is not None:
        advantages = (rewards - mu) / denom  # (B,G)
    else:
        advantages = rewards - mu  # (B,G)

    #advantages = advantages.view(-1)  # (B*G,),分组计算完毕得到优势值后展平
    advantages = advantages.reshape(-1)  # (B*G,)
    metadata = {
        "mean_reward": float(raw_rewards.mean()),
        "std_reward": float(raw_rewards.std()),
        "max_reward": float(raw_rewards.max()),
        "min_reward": float(raw_rewards.min()),
        "fraction_zero_advantage": float((advantages == 0).float().mean()),
    }
    return advantages, metadata



"""
要更新参数，关键是要得到策略梯度的估计器，即∇θJ = A * ∇θ log πθ(y_t|...)

所以我们要先计算这个J，然后对 J 求导，得到策略梯度的估计器。

对于on-policy 的GRPO，可以先求 A * log πθ ，把 J 看成 A * log πθ 的函数，再对 A * log πθ 求导，得到 A * ∇log πθ，即策略梯度的估计器。然后就可以用这个估计器来更新参数了。

对于off-policy的GRPO，因为轨迹是旧策略的，而在新策略下，这条轨迹的概率不一样了，所以要进行修正。
修正需要用到重要性加权，即 w_t = πθ(y_t|x,y_<t) / π⁰(y_t|x,y_<t)。其中 πθ 是新策略，π⁰ 是旧策略。
所以此时策略梯度的估计器是 A * w_t * ∇θ log πθ(y_t|...)
也就是说：此时∇θJ = A * w_t * ∇log πθ

与前一个一样，我们也需要得到J，他的导数就是上面这个策略梯度的估计器。

这个变量J，也就是替代目标为 A * w_t，不是 A * log πθ。对 A * w_t 求导，log πθ 会从 w_t 的导数里重新出现。

w_t = πθ(y_t|x,y_<t) / π⁰(y_t|x,y_<t)
    = exp( log πθ(y_t|...) - log π⁰(y_t|...) )
off-policy 的替代目标是 A * w_t，不是 A * log πθ。对 A * w_t 求导，log πθ 会从 w_t 的导数里重新出现。

推导一遍
off-policy 想要的梯度估计是
g_t = A * w_t * ∇θ log πθ(y_t|...)
现在问:哪个标量函数求导能得到它？

注意 w_t 对 θ 的导数

∇θ w_t = ∇θ exp(log πθ - log π⁰)
       = exp(log πθ - log π⁰) * ∇θ(log πθ - log π⁰)
       = w_t * ∇θ log πθ          # log π⁰ 是常数，导数为 0
所以

∇θ (A * w_t) = A * w_t * ∇θ log πθ = g_t
正好就是想要的估计器。因此损失写成

loss_t = - A * w_t
autograd 自动求导时会得到 -A * w_t * ∇log πθ，梯度下降就等价于对策略梯度做上升。

和 on-policy 对照
情形	     替代目标 J	                          梯度
on-policy	A * log πθ	                      A * ∇log πθ
off-policy	A * w_t，其中 w_t = πθ/π⁰	    A * w_t * ∇log πθ
on-policy 时没有旧策略修正，直接对 log πθ 求导就够了。

off-policy 时，w_t 本身就是 πθ 的函数；对 w_t 求导，自然产生 w_t * ∇log πθ。所以目标函数从"log π"变成了"πθ/π⁰"，但梯度里的 score function 一点没少。

如果把旧策略固定住，在更新开始时 πθ = π⁰，那么 w_t = 1，梯度退化为 A * ∇log πθ，和 on-policy 一致。

为什么不能写成 -A * w_t * log πθ
因为对 A * w_t * log πθ 求导会多出一项

∇(A * w * log π) = A * w * ∇log π  +  A * log π * ∇w
                 = A * w * ∇log π  +  A * log π * w * ∇log π
第二项是多余的。w 和 log π 都依赖 θ，不能同时让它们参与求导。


同时，由于旧策略和新策略对同一个 token 的概率可能相差较大，重要性权重 w_t 也可能远大于 1 或远小于 1。这会带来两个问题。

第一，w_t 很大会把该 token 的梯度成倍放大，少数"过时但比值极端"的样本会主导整个 batch 的更新，估计器方差急剧上升；极端情况下，一步更新就把策略推离旧策略很远，训练震荡甚至发散。第二，w_t 很小时，该 token 几乎不贡献梯度，旧样本被浪费，有效样本量下降。更麻烦的是，随着同一批 rollout 被反复用于多个梯度步，πθ 与 π⁰ 的差距会不断累积，token 级重加权的偏差也会随之变大——它只修正了当前 token，前后缀仍然来自旧策略。

如果直接用整条轨迹的概率乘积作为权重，问题会更严重:序列级权重是 L 个比值的乘积，方差随回答长度指数增长。PPO/GRPO 改用 token 级比值，正是为了避开这种指数爆炸；但 token 级比值本身仍可能出现极端值。

因此，不能对重要性权重不加约束地使用。PPO/GRPO 的做法是把权重限制在旧策略附近的一个信任区间内，这就是 clip

J_clip = min( A  w_t,  A  clip(w_t, 1-ε, 1+ε) )

其中：clip(w, 1-ε, 1+ε) = min(max(w, 1-ε), 1+ε)
作用是：把 w_t 限制在 1-ε 1+ε内，避免 w_t 过大或过小，导致梯度估计器的方差急剧上升。
w < L        -> L
L <= w <= U  -> w
w > U        -> U

分情况看

A >= 0 时，J = A  min(w_t, 1+ε)。w_t 没超过 1+ε，就正常按 w_t 放大梯度；一旦超过，min 取常数 1+ε，梯度变为 0，不再继续抬高这个 token 的概率。
A < 0 时，J = A * max(w_t, 1-ε)。w_t 没低于 1-ε，就正常压低；一旦低于，梯度变为 0，不再继续打压。

从优化角度看，min 是在两个候选目标里取更悲观的那个:不允许模型通过把权重推到信任区间外来"虚报"更高的目标值。超出区间后目标变成常数，梯度为 0。

"""
"""
w_t = πθ(y_t|...) / π⁰(y_t|...)
w_t > 1:新策略比旧策略更倾向这个 token；
w_t < 1:新策略比旧策略更不倾向这个 token。
优化器想往哪走
A > 0(这个 token 带来正优势):更新想提高它的概率，也就是让 w 变大；
A < 0(负优势):更新想降低它的概率，也就是让 w 变小。
裁剪的目的，是阻止策略在"优势鼓励的方向"上离旧策略太远。所以

A >= 0，危险方向是 w 一直往上冲 → 需要卡上界；
A < 0，危险方向是 w 一直往下掉 → 需要卡下界。
反方向不需要卡，因为那是往旧策略回靠，不会让策略跑得更远。
所以更准确的说法是

A >= 0: w 的有效约束是上界 1+ε
A < 0 : w 的有效约束是下界 1-ε

一张表
优势	更新想做的事	危险方向	真正被裁的一侧	  简化后的目标
A >= 0	提高 w	        w 变大	     上界 1+ε	    A * min(w, 1+ε)
A < 0	降低 w	        w 变小	     下界 1-ε	    A * max(w, 1-ε)
直觉总结
PPO 的 clip 不是"把 w 关进一个对称的笼子"，而是只拦住继续远离旧策略的那一步

好 token(A>0)已经比旧策略更受偏爱超过 1+ε 倍，就别再继续加码；
坏 token(A<0)已经比旧策略更被压制超过 1-ε 倍，就别再继续打压；
反过来，如果好 token 的概率暂时比旧策略低，或者坏 token 的概率暂时比旧策略高，更新想把它拉回旧策略附近，这种方向是允许的，不需要裁。
"""
"""
其中 ε 就是 cliprange，常见取值 0.1 或 0.2。它的作用不是让目标函数更"正确"，而是一个偏差-方差的折中:用一点偏差换取方差可控和有界更新，使同一个 rollout 批次可以安全地走多个梯度步。

ε 的大小也有取舍

ε 太小，策略几乎不能偏离旧策略，学习变慢，大量更新被裁掉；
ε 太大，过时样本的权重仍可能主导更新，稳定性下降。

实践中通常同时监控平均/最大重要性权重、clip 比例和近似 KL。如果比值持续偏离 1，说明策略跑得太快，应减小每个 rollout 批次的训练步数或 ε。

GSPO 则把同样的思想搬到序列级:先算整条回答的几何平均权重

s = exp( (1/L)  Σ_t log(πθ/π⁰) )

再对 s 做 clip。这样整个序列共享一个比值，避免 token 级比值在长序列上累积出极端值；代价是几何平均和裁剪本身也引入偏差。

---

改动点只有一个:你原来的"同一条轨迹的概率相差"建议改成"同一个 token 的概率相差",因为 w_t 是 token 级的；整条轨迹的比率是序列级方法(如 GSPO)才用的。

"""

"""
这个函数只负责算逐 token 的替代损失，不做最终的 mask 聚合；聚合由后面的 aggregate_loss_across_microbatch 完成
"""
def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor, # 每个 token 的对数概率
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None, # 旧策略的 log 概率
    cliprange: float | None = None,  # PPO 的 ε
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算逐 token 的替代损失
    raw_rewards_or_advantages: 每条轨迹(prompt+回答)的原始奖励或优势，形状为 (batch_size, 1) 或 (batch_size, ).作为一个逐序列的系数乘到 token 上，所以形状必须能广播到 (B, L)。通常是先 unsqueeze(-1) 变成 (B,1) 再乘。

    policy_log_probs: 当前策略 πθ 在实际采样到的 token 上的逐 token 对数概率，形状为 (batch_size, seq_len)

    importance_reweighting_method: 决定用哪种梯度估计器，可选值为 "none", "noclip", "grpo", "gspo"
    一张对比表
方法	      权重粒度	                     公式	
none	         无	                       -A*logp	    on-policy
noclip	      token 级	                   -A*w_t	
grpo	    token 级 + clip	         -min(A*w_t, A*clip(w_t))	
gspo	  序列级几何平均 + clip	        -min(A*s, A*clip(s))	
# ============ importance_reweighting_method 四种方法 ============
#
# 【none】不做重要性重加权(仅 on-policy)
#   逐 token 损失 : -A*logp
#   必需参数      : 无
#   偏差/方差     : 无修正偏差；方差来自采样
#
# 【noclip】token 级重要性重加权，不裁剪
#   逐 token 损失 : -A*w_t
#   必需参数      : old_log_probs
#   偏差/方差     : 方差比序列级低；策略偏离旧策略后有偏
#
# 【grpo】token 级重要性重加权 + PPO 裁剪
#   逐 token 损失 : -min(A*w_t, A*clip(w_t, 1-ε, 1+ε))
#   必需参数      : old_log_probs, cliprange
#   偏差/方差     : 用少量偏差换稳定性和有界更新
#
# 【gspo】序列级几何平均权重 + 裁剪
#   逐 token 损失 : -min(A*s, A*clip(s, 1-ε, 1+ε))
#   必需参数      : old_log_probs, cliprange, response_mask
#   偏差/方差     : 长序列更稳；几何平均和裁剪仍有偏
#
# 符号:
#   w_t = exp(logp_t - old_logp_t)
#   s   = exp(mean_{t∈response}(logp_t - old_logp_t))
#   clip(x, 1-ε, 1+ε) = min(max(x, 1-ε), 1+ε)
# ================================================================

    old_log_probs: 旧策略的 log 概率，形状为 (batch_size, seq_len)
    
    cliprange: 类型:float | None，就是 PPO 的 ε。
    含义:重要性权重的信任区间 [1-ε, 1+ε]。只有 "grpo" 和 "gspo" 需要；"none"/"noclip" 下即使传入也不起作用
    
    response_mask: 形状:(B, L)，和 labels / policy_log_probs 对齐；通常是 bool，response token 位置为 True，prompt 和 padding 为 False。
    作用: 对 "gspo" 是必需的:序列级权重是 response token 上 log 比值的平均

    return:
    per_token_policy_gradient_loss:(B, L)，逐 token 的负替代目标。之所以取负，是因为 PyTorch 优化器做梯度下降，而我们要最大化策略目标。

    metadata:dict，用于日志，例如平均/最大重要性权重、clip 比例等。它不影响梯度。


GRPO/PPO 用的是 token 级重要性权重

w_t = πθ(y_t|x,y_<t) / π⁰(y_t|x,y_<t)
它有两个问题

方差比序列级乘积小，但不是对整条轨迹概率比的正确修正；它只修正当前 token，前缀和后缀仍来自旧策略，因此有偏。
每个 token 独立裁剪，长回答里 token 一多，噪声和裁剪行为会变得不稳定；对 MoE 模型，路由变化还会让 token 级比值抖动更大。
而"真正正确"的序列级权重是

code.txt
w_seq = πθ(y|x) / π⁰(y|x) = Π_t w_t （累乘）
它是无偏的，但方差随回答长度指数增长，长 CoT 直接爆炸，实际不可用。

GSPO 取中间路线:对 token 级比值取几何平均

s = ( Π_t w_t ) ^ (1/L)
  = exp( (1/L) * Σ_t log(w_t) )
这样既保留了"整条序列一个权重"的结构，又不会像连乘那样指数爆炸。分母 L 是 response token 数，prompt 和 padding 不参与。

    """
    # raw_rewards_or_advantages 是(B,),含义是每一个轨迹(prompt+回答)的原始奖励或优势，也就是用来计算J的A或者R
    # policy_log_probs 是(B,L)，是之前的函数计算得到的每一个token的log p值，每一行就是一个轨迹，每行的第j列就是在前j个token的条件下，第j+1个token的log p值
    # 也就是说每一行就是一个轨迹的token的log p值
    advantages = raw_rewards_or_advantages
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)      # (B,) → (B,1)，广播到 (B,T)
    if importance_reweighting_method == "none":
        # none，on-policy，不做重要性重加权，直接就是-A*logp
        per_token_policy_gradient_loss = -advantages * policy_log_probs

    elif importance_reweighting_method == "noclip":
        # noclip，token 级重要性重加权，off-policy，不裁剪，但是需要 old_log_probs，用来计算 w_t，公式为:-A*w_t，其中 w_t = exp(logp_t - old_logp_t)
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for off-policy importance reweighting")
        # 计算 w_t
        w_t = torch.exp(policy_log_probs - old_log_probs.detach())  # 不 detach w_t：∇w_t = w_t·∇logp_t，梯度经 policy_log_probs 流动
        # 计算 -A*w_t
        per_token_policy_gradient_loss = -advantages * w_t

    elif importance_reweighting_method == "grpo":
        # grpo,是token 级重要性重加权 + PPO 裁剪，公式为:-min(A*w_t, A*clip(w_t, 1-ε, 1+ε))，其中 w_t = exp(logp_t - old_logp_t)
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for off-policy importance reweighting")
        if cliprange is None:
            raise ValueError("cliprange is required for grpo importance reweighting")
        # 计算 w_t
        w_t = torch.exp(policy_log_probs - old_log_probs.detach())  # 不 detach w_t：∇w_t = w_t·∇logp_t，梯度经 policy_log_probs 流动
        # 计算 -min(A*w_t, A*clip(w_t, 1-ε, 1+ε))
        clipped_ratio = torch.clamp(w_t, 1 - cliprange, 1 + cliprange)
        unclipped = advantages * w_t
        clipped = advantages * clipped_ratio
        per_token_policy_gradient_loss = -torch.minimum(unclipped, clipped)

    elif importance_reweighting_method == "gspo":
        # gspo,序列级几何平均权重 + 裁剪，公式为:-min(A*s, A*clip(s, 1-ε, 1+ε))，其中 s = exp(mean_{t∈response}(logp_t - old_logp_t))
        # GSPO(Group Sequence Policy Optimization) 是 GRPO 的序列级版本:它不再给每个 token 单独算重要性权重，而是把整条回答的所有 token 比值合成一个几何平均序列权重 s，再用 PPO 式 clip 约束这个序列权重。
        if old_log_probs is None:
            raise ValueError(
                "old_log_probs is required for off-policy importance reweighting"
            )
        if cliprange is None:
            raise ValueError("cliprange is required for gspo importance reweighting")
        if response_mask is None:
            raise ValueError("response_mask is required for gspo importance reweighting")
 
        # 逐 token 的 log 比值；旧策略是常数，不参与求导
        log_ratio = policy_log_probs - old_log_probs.detach()
 
        # 只在 response token 上取平均，prompt/padding 不进入分母
        mask = response_mask.to(dtype=log_ratio.dtype)
        response_len = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        log_s = (log_ratio * mask).sum(dim=-1, keepdim=True) / response_len
 
        # 序列级几何平均权重 s，形状 (B,1)
        s = torch.exp(log_s)
 
        # 序列级 PPO 裁剪
        s_clipped = torch.clamp(s, 1 - cliprange, 1 + cliprange)
        unclipped = advantages * s
        clipped = advantages * s_clipped
        per_sequence_loss = -torch.minimum(unclipped, clipped)      # (B,1)
 
        # 同一条序列的每个 token 共享同一个损失值
        per_token_policy_gradient_loss = per_sequence_loss.expand_as(policy_log_probs)
 
    else:
        raise ValueError(
            f"Unknown importance_reweighting_method: {importance_reweighting_method}"
        )
 
    return per_token_policy_gradient_loss, {}






"""
compute_policy_gradient_loss        -> 每个 token 一个损失值，形状 (B,L)
aggregate_loss_across_microbatch    -> 用 response_mask 选出回答 token，
                                       再按 sequence 或 constant 归一化成标量
"""
"""
这个函数位于 GRPO 损失计算的**最后一步**：它把上游算出的逐 token 损失矩阵压缩成一个标量，供 `loss.backward()` 使用。下面详细解释。
## 在整个流程中的位置
```
tokenize → get_response_log_probs → compute_policy_gradient_loss
                                          ↓ 产出 (batch, seq_len) 的逐token损失
                              aggregate_loss_across_microbatch   ← 这个函数
                                          ↓ 产出标量 loss
                              grpo_train_step 里 loss.backward()
```

策略梯度估计器（作业公式 28）本质是对**所有 token 求和**，但 PyTorch 优化器做的是梯度下降，所以上游 `compute_policy_gradient_loss` 返回的是取负后的逐 token 损失 `-Â_t · log π_θ(y_t|…)`（shape `(batch_size, seq_len)`），或者是-A_t*w_t，其中 prompt/padding 位置的值还没被排除。这个函数要做的就是把矩阵 → 标量，**同时决定“如何加权”**——这个加权方式直接对应不同的 RL 算法变体。

## 各参数详解
### `per_token_policy_gradient_loss`
- **shape**: `(batch_size, sequence_length)`，来自 `compute_policy_gradient_loss` 的返回值。
- **含义**： 每个位置(token)一个损失值。对响应 token，它是 `-优势 × log概率`（已取负）；对非响应位置（prompt/padding），值通常也是数（没被置零），所以**必须靠 mask 排除**，不能直接 `.mean()`。
- **梯度**： 这里的元素都连着计算图，聚合必须用可微操作（乘 mask、求和、除法），这样 backward 才能把梯度传回模型权重。

### `mask`
- **shape**: `(batch_size, sequence_length)`，与损失矩阵逐位对齐，即 `tokenize_prompt_and_output` 产出的 `response_mask`。
- **取值**： 1 = 该位置的 label 是真正的响应 token；0 = prompt 或 padding。
- **两个作用**：
  1. **分子**：用 `loss * mask` 把非响应位置的损失清零，不参与求和；
  2. **分母**：归一化时分母也用 mask 计数（如 `mask.sum()`），保证平均是“对有效 token 的平均”，而不是对全部位置（包括 padding）的平均。
- 注意它是 0/1 浮点张量，直接相乘即可，`0 × loss = 0` 的位置梯度自然也是 0。

### `loss_normalization` —— 两种聚合方案（这是本函数的核心）

**`"sequence"`（默认，标准 GRPO）**：先每条序列内平均，再跨序列平均。

```text
loss = mean_over_sequences( sum_t(loss[i,t] * mask[i,t]) / sum_t(mask[i,t]) )
```

对应作业公式 (27)(28)：每条序列先除以自己的长度 `len(y)`，再对 B·G 条序列取平均。效果是**每条序列对梯度的贡献相等，序列越长，其内部每个 token 的权重越小**（长序列里的 token 被降权 1/len 倍）。

**`"constant"`（Dr. GRPO 变体，§5.1）**：所有被 mask 的 token 损失直接求和，再除以一个**固定常数**。

```text
loss = sum_{i,t}(loss[i,t] * mask[i,t]) / normalization_constant
```

对应 Dr. GRPO 的公式 (33)：常数 `Z = B·G·L`（批量大小 × 最大生成长度，如 512）。效果是**所有 token 等权**（权重 1/Z），与序列长短无关。

**为什么要区分这两种**？Dr. GRPO 论文指出 sequence 模式改变了梯度估计器的期望：它系统性压低长序列中正确 token 的贡献，导致模型偏好短回答（比如推理任务中生成长度偏短的偏差）。constant 模式把估计器“还原”为无偏的策略梯度，只是用一个与训练全程固定的常数控制损失的数值尺度（防止 sum 随 batch 变大而爆炸）。§5.4 的四个变体（GRPO_constant / Dr_GRPO / RFT / MaxRL）全部用 `constant`，只有标准 GRPO 用 `sequence`。

### `normalization_constant`
- **含义**： 就是 Dr. GRPO 里的常数归一化器 `Z`，整个训练期间固定不变。
- **取值惯例**： `rollout_batch_size × 最大生成长度`（作业里建议 512），也可以理解成“每个 microbatch 最多有多少个有效 token”。
- **约束**： 仅当 `loss_normalization = "constant"` 时使用且**必填**（为 `None` 时应抛错）；`"sequence"` 模式下完全忽略它。测试里传的是 `normalization_constant=42`。

### 返回值
- 一个**标量（0 维）张量**，且必须保留计算图——`grpo_train_step` 会拿它做 `backward()`（梯度累积时还会先除以 microbatch 数）。

## 实现要点提示

1. `"sequence"` 模式：分子 `sum(loss * mask, dim=seq)`，分母 `sum(mask, dim=seq)`，逐序列相除后再 `.mean()`。分母为 0（该序列没有响应 token）时可加一个极小 eps 防止 NaN。
2. `"constant"` 模式：`sum(loss * mask) / normalization_constant`；`normalization_constant is None` 时抛 `ValueError`/`NotImplementedError`。
3. 千万不要先对矩阵做 `.mean()` 再乘 mask——那会把 prompt/padding 位置也算进分母，归一化就错了。正确顺序永远是“**先 mask、再求和、后归一化**”。


"""
def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """
    对微批量内的 token 损失进行聚合，根据 loss_normalization 决定聚合方式。

    per_token_policy_gradient_loss: (batch_size, sequence_length) ，每个位置(token)一个损失值，来自 `compute_policy_gradient_loss` 的返回值。分为四种情况：
    1.none，也就是默认on-policy，是每个token的-A*logp_t
    2.noclip，是每个token的-A*w_t
    3.grpo，有重要性重加权和clip，公式为:-min(A*w_t, A*clip(w_t, 1-ε, 1+ε))，其中 w_t = exp(logp_t - old_logp_t)

    mask: (batch_size, sequence_length) ，与 per_token_policy_gradient_loss 形状相同，与损失矩阵逐位对齐，即 `tokenize_prompt_and_output` 产出的 `response_mask`。标注了哪些位置是响应 token，哪些是 prompt/padding。从prompt的最后一个token开始，向后扩展，直到padding。

    loss_normalization: 聚合方案，"sequence" 或 "constant"。默认 "sequence"，也就是标准 GRPO。 constant 模式需要 normalization_constant，也就是 Dr. GRPO 变体的常数归一化器。把所有被 mask 的 token 损失直接求和，再除以 normalization_constant。

    normalization_constant: Dr. GRPO 变体的常数归一化器，Z = B·G·L（批量大小 × 最大生成长度，如 512）。仅当 loss_normalization = "constant" 时使用且**必填**（为 None 时应抛错）；`"sequence"` 模式下完全忽略它。
    """
    if loss_normalization == "sequence":
        # 先把prompt/padding位置的损失清零，不参与求和,仍是(B,L)
        per_token_loss=per_token_policy_gradient_loss * mask
        # 再序列内求和,维度变换为：(B,L)->(B,)
        per_token_loss=per_token_loss.sum(dim=-1)
        # 每条序列除以自己的响应 token 数（不是序列全长，prompt/padding 不计）
        per_token_loss=per_token_loss / (mask.sum(dim=-1) + 1e-8)
        # 最后跨序列取平均，也就是除以公式里面的B·G,这里的B指的是有多少个prompt，G是每组采样多少个回答，输入的per_token_policy_gradient_loss是(B,L),但是第一维的大小其实就是公式里面的B·G了
        loss=per_token_loss.mean()
    elif loss_normalization == "constant":
        # Dr. GRPO / RFT / MaxRL：所有入选 token 求和后除以固定常数 Z
        # Z = B·G·L（批量大小 × 最大生成长度，如 512）
        if normalization_constant is None:
            raise ValueError("normalization_constant is required for constant normalization")
        #x.sum()(没传dim）torch.Size([])，0维标量对所有元素求和，把所有维度都压掉
        # x.sum(dim=-1)	(B,)	只沿最后一维求和，保留行
        # x.sum(dim=0)	(L,)	沿第一维求和
        # 所以这里返回的也是0维标量
        # 全部求和再除Z,也相当于先对每条序列除固定的一个最大序列长度L，再对所有序列求和，然后跨序列取平均，也就是除B*G
        # 而Z就是B·G·L，所以这里除以Z就是除以B*G*L
        """
把两种 loss 归一化放在一起看，结构其实是同一个——“逐序列归一化 → 跨 B·G 条序列平均"，唯一区别是除数：

模式	                      逐序列除数	            每个响应 token 的最终权重
sequence（标准 GRPO）	该序列自己的响应长度 len(y_i)（变长）	1/(B·G·len(y_i)) —— 序列越长，token 权重越小
constant（Dr. GRPO）	固定的最大生成长度 L（人人都一样）	1/(B·G·L) —— 所有 token 等权
这正是作业 think_about_length_normalization 思考题的核心：除以自身长度会让长回答里的每个 token 被系统性降权，模型因此偏向短回答；除以固定 L 消除了这个长度偏差。你现在这个分解恰好把“为什么叫常数归一化”看得一清二楚——除数 L 与序列无关。
        """
        loss = (per_token_policy_gradient_loss * mask).sum() / normalization_constant
    else:
        raise NotImplementedError(f"Unsupported loss_normalization: {loss_normalization}")
    return loss


"""
GRPO 训练步代码
"""
def grpo_train_step(
    model, tokenizer, optimizer, gradient_accumulation_steps, max_grad_norm,
    reward_fn, repeated_prompts, rollout_responses, repeated_ground_truths, group_size,
    baseline="mean", advantage_eps=1e-6, advantage_normalizer="std",
    importance_reweighting_method="none", old_log_probs=None, cliprange=None,
    loss_normalization="sequence", normalization_constant=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """
    GRPO 训练的单批次更新。
    输入一批【已经采样好的】rollout（生成发生在 vLLM 侧，不在这里），
    完成"打分 -> 组归一化优势 -> 切 microbatch 前向/反向 -> 梯度裁剪 -> optimizer.step() -> 清零梯度"。

    ==================== 执行流程 ====================
    (1) tokenize:   tokenize_prompt_and_output(repeated_prompts, rollout_responses)
                    -> input_ids / labels / response_mask，形状 (B*G, L)
    (2) 前向:       get_response_log_probs(model, input_ids, labels)
                    得到当前策略 πθ 的逐 token log 概率（带梯度）
    (3) 打分:       compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
                    -> raw_rewards (B*G,)，未归一化的序列级奖励
    (4) 优势:       compute_group_normalized_rewards(raw_rewards, group_size, baseline,
                    advantage_eps, advantage_normalizer) -> advantages (B*G,)
    (5) 损失:       compute_policy_gradient_loss(advantages, policy_log_probs,
                    importance_reweighting_method, old_log_probs, cliprange, response_mask)
                    -> 逐 token 损失 (B*G, L)
    (6) 累积:       全批 B*G 行切成 gradient_accumulation_steps 个 microbatch，逐个:
                    aggregate_loss_across_microbatch(...) 得到标量 loss_m，
                    loss_m / gradient_accumulation_steps 后 backward()，梯度自动累加
    (7) 裁剪:       max_grad_norm 不为 None 时，optimizer.step() 前对全部参数
                    clip_grad_norm_（metadata 记录裁剪前的梯度范数）
    (8) 更新:       optimizer.step()，然后 optimizer.zero_grad(set_to_none=True)
                    （测试断言返回时所有 param.grad is None，所以必须在 step 后清零）
    (9) 返回:       (loss, metadata)

    ==================== 参数说明 ====================

    model: 被训练的策略模型 πθ。
        注意它【不做生成】——rollout 已由 vLLM 采样好作为 rollout_responses 传入；
        model 在这里只负责前向计算 log probs（带梯度）以及被 optimizer 更新。

    tokenizer: 用于把 prompt / response 字符串编码成 token id 的分词器。

    optimizer: 模型的优化器（如 torch.optim.AdamW），由调用方创建并传入。

    gradient_accumulation_steps: 梯度累积步数 k。
        把 rollout_batch_size = B*G 条序列切成 k 个 microbatch（尽量等大），
        每个 microbatch 各自 forward + backward，loss 除以 k 后梯度累加，
        等价于一个大 batch 的更新，只是省显存。
        例：测试里 8 条序列、k=2 -> 每个microbatch 4 条。

    max_grad_norm: 梯度范数裁剪阈值，None 表示不裁剪。
        非 None 时在 optimizer.step() 之前对全部参数做 clip_grad_norm_。

    reward_fn: 奖励打分器，签名固定为 reward_fn(response, ground_truth) -> dict。
        注意打分对象是【整条回答字符串】，不是逐 token！
        返回字典固定三个键：
            "reward"        总奖励（本作业 = answer_reward，0/1，无部分分）
            "format_reward" 格式奖励：是否满足 <think>…</think><answer>…</answer> 结构
            "answer_reward" 答案奖励：解析出的答案是否与 ground truth 等价
        优势是序列级标量，之后广播（unsqueeze 成 (B*G,1)）到该序列的所有响应 token 上。

    repeated_prompts: 长度 = B*G 的 prompt 列表。
        每个 prompt 连续重复 group_size 次（prompt-major 排布），
        第 i 个 prompt 占据行 [i*G, (i+1)*G)。组归一化依赖这个约定。

    rollout_responses: 长度 = B*G 的模型采样回答列表，与 repeated_prompts 逐行对应。

    repeated_ground_truths: 长度 = B*G 的标准答案列表，与 rollout_responses 逐行对应
        （每个 prompt 的答案连续重复 group_size 次，所以叫 "repeated"）。

    group_size: G，每个 prompt 采样的回答数（组大小）。

    baseline: 优势的组内基线。可选：
        "mean" - A_ij = r_ij - μ_i，减去组内奖励均值（标准 GRPO / Dr.GRPO / MaxRL 都用它）
        "none" - A_ij = r_ij，直接用原始奖励（RFT 用它）

    advantage_eps: 防除零小常数，默认 1e-6，【加在归一化分母上】：
            A_ij = (r_ij - baseline) / (normalizer + eps)
        仅 normalizer 为 "std" 或 "mean" 时起作用；normalizer="none" 时不除，eps 无效。

    advantage_normalizer: 组内归一化方式。可选：
        "std"  - 除以组内标准差 std_i（用 torch.std 默认的无偏版本）——标准 GRPO
        "none" - 不归一化——Dr.GRPO（baseline="mean"）或 RFT（baseline="none"）
        "mean" - 除以组内均值 μ_i——MaxRL
        对比：std 归一化不再保持梯度估计器的期望（是稳定性技巧，让每组更新范数大致相等）；
              Dr.GRPO 论文主张去掉它（配合 constant loss 归一化）。

    importance_reweighting_method: 重要性重加权方式（off-policy 支持）。四种：
        方法      权重粒度            逐 token 损失                              必需参数
        none      无(on-policy)       -A * logp_t                                无
        noclip    token 级            -A * w_t                                   old_log_probs
        grpo      token 级 + clip     -min(A*w_t, A*clip(w_t, 1±ε))              old_log_probs, cliprange
        gspo      序列级几何平均+clip  -min(A*s,   A*clip(s,   1±ε)) 按token广播   old_log_probs, cliprange, response_mask
        其中 w_t = exp(logp_t - old_logp_t)（token 级比值），
             s   = exp(mean_{t∈response}(logp_t - old_logp_t))（序列级几何平均比值）。

    old_log_probs: (B*G, L)，旧策略 π⁰（采样这条 rollout 时的模型）的逐 token log 概率。
        在训练脚本里、模型更新【之前】算好并 detach 传入。
        importance_reweighting_method="none" 时忽略；其余三种必须提供。

    cliprange: PPO 裁剪强度 ε（如 0.1 / 0.2），重要性权重被裁剪到 [1-ε, 1+ε]。
        仅 "grpo" / "gspo" 需要；"none" / "noclip" 下即使传入也不起作用。

    loss_normalization: microbatch 内损失聚合方式：
        "sequence" - 标准 GRPO：每条序列先除以自己的响应 token 数（组内平均），
                     再跨序列平均（除以 microbatch 行数）。
                     长序列里的 token 被降权 1/len(y) 倍 -> 模型偏向短回答（长度偏差）。
        "constant" - Dr.GRPO：所有被 mask 的 token 损失直接求和，除以固定常数 Z。
                     所有 token 等权 1/Z，消除长度偏差。

    normalization_constant: 常数归一化器 Z = B*G*L
        （rollout_batch_size × 最大生成长度，作业里 L=512；整个训练期间固定）。
        仅 loss_normalization="constant" 时使用且必填（为 None 应抛错）；
        "sequence" 模式下完全忽略。

    ==================== 返回值 ====================
    (loss, metadata):
    loss: 标量张量。本批损失，已按梯度累积调整（每个 microbatch 的 loss_m / k 之和，供日志）。
    metadata: dict，日志用统计，测试不校验内容。建议包含：
        各 microbatch 的 loss、裁剪前梯度范数（adapter 明确要求）、
        优势/奖励统计（mean_reward、mean_std 等）、clip 发生比例等。

    ==================== 常用变体组合（对应测试） ====================
    标准GRPO:        baseline="mean", normalizer="std",  loss="sequence"
    GRPO_constant:   baseline="mean", normalizer="std",  loss="constant", Z=B*G*L
    Dr_GRPO:         baseline="mean", normalizer="none", loss="constant"
    RFT:             baseline="none", normalizer="none", loss="constant"
    MaxRL:           baseline="mean", normalizer="mean", loss="constant"
    off-policy:      method="noclip"/"grpo"/"gspo" + old_log_probs (+cliprange)

    注意：作业 §5.4 建议跳过零优势序列以加速训练（RFT 等变体会产生大量零优势），
    但这是否会影响 rft 变体的快照匹配，以测试运行结果为准。
    """
    device = next(model.parameters()).device

    # 1) 计算奖励 + 优势
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths)
    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline=baseline,
        advantage_eps=advantage_eps, advantage_normalizer=advantage_normalizer)

    # 2) 剪掉零优势序列，因为优势是0，计算loss时就是0*logP，没有贡献，除去避免计算
    keep = advantages != 0 # 这里得到的keep是一个bool张量，True表示保留，False表示剪掉，作索引操作时会自动过滤掉False对应的元素，取出True对应的行
    kept_advantages = advantages[keep]
    # 用for循环和直接repeat_prompts[keep]效果相同，只是for循环更通用
    kept_prompts = [p for p, k in zip(repeated_prompts, keep.tolist()) if k]
    kept_responses = [r for r, k in zip(rollout_responses, keep.tolist()) if k]
    kept_old_log_probs = old_log_probs[keep] if old_log_probs is not None else None

    metadata: dict[str, torch.Tensor | float] = {}
    metadata.update(reward_metadata)
    metadata.update(advantage_metadata)

    n_total = len(repeated_prompts)        # ← 剪枝前的总序列数
    n_kept = len(kept_prompts)
    if n_kept == 0:                        # 整批无信号：不更新
        metadata["gradient_norm"] = 0.0
        return torch.tensor(0.0), metadata

    # 3) tokenize + 微批次累积
    batch = tokenize_prompt_and_output(kept_prompts, kept_responses, tokenizer)
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    response_mask = batch["response_mask"].to(device)


    microbatch_size = max(1, math.ceil(n_total / gradient_accumulation_steps))

    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = torch.tensor(0.0)
    entropy_sum = 0.0

    # 除去零优势序列后还有n_kept条序列，按照microbatch_size进行微批次累积
    for start in range(0, n_kept, microbatch_size):
        # 从start开始到end，取microbatch_size条序列，进行微批次训练
        end = min(start + microbatch_size, n_kept)
        # 得到当前微批次的输入_ids、标签、响应掩码，用于计算损失
        microbatch_input_ids = input_ids[start:end]
        microbatch_labels = labels[start:end]
        microbatch_response_mask = response_mask[start:end]
        # 先计算当前微批次的log_probs
        out = get_response_log_probs(model, microbatch_input_ids, microbatch_labels,return_token_entropy=True)
        now_log_probs = out["log_probs"]
        now_entropy = out["token_entropy"]
        # 看是否需要用到old_log_probs
        if old_log_probs is not None:
            now_old_log_probs = kept_old_log_probs[start:end].to(device)
        else:
            now_old_log_probs = None
        # 计算当前微批次的损失
        # 首先计算逐token的损失
        now_per_token_loss, loss_metadata=compute_policy_gradient_loss(
            kept_advantages[start:end],now_log_probs,             importance_reweighting_method=importance_reweighting_method,
            old_log_probs=now_old_log_probs, cliprange=cliprange,
            response_mask=microbatch_response_mask)
        # 再计算序列的损失
        now_seq_loss=aggregate_loss_across_microbatch(
            now_per_token_loss, microbatch_response_mask, loss_normalization, normalization_constant)
        # 加权
        if loss_normalization == "sequence":
            now_seq_loss = now_seq_loss * ((end - start) / n_total)   # ← 序列归一化才加权！

        # 熵（日志）：响应 token 平均，再按序列占比加权累加
        weight = (end - start) / n_total
        entropy_sum += ((now_entropy * microbatch_response_mask).sum()
                        / microbatch_response_mask.sum().clamp(min=1)).item() * weight

        for key, value in loss_metadata.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metadata.setdefault(f"mb0_{key}", value.detach())
        accumulated_loss = accumulated_loss + now_seq_loss.detach()
        now_seq_loss.backward()

    # 4) 裁剪 + 步进 + 清零（clip_grad_norm_ 的返回值恰是裁剪前的全局范数）
    if max_grad_norm is not None:
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        gradient_norm = torch.norm(torch.stack(
            [p.grad.norm() for p in model.parameters() if p.grad is not None]))
    metadata["gradient_norm"] = float(gradient_norm)
    metadata["mean_token_entropy"] = entropy_sum
    metadata["n_sequences_used"] = n_kept

    # 顺序很重要：先 clip（记录裁剪前范数）→ 再 step → 最后清零（测试断言 grad 全为 None）
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return accumulated_loss, metadata

