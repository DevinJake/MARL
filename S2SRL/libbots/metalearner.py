import torch
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
class MetaLearner(object):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes 
    (before and after the one-step adaptation), compute the inner loss, compute 
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, 
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized 
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, 
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """
    def __init__(self, net, device='cpu', beg_token=None, train_data_support=None):
        self.net = net
        self.device = device
        self.beg_token = beg_token
        # The training data from which the top-N samples (support set) are found.
        self.train_data_support = train_data_support

    def establish_support_set(self, train_data):
        # Find top-N in train_data_support;
        print()


    # NOTE 概率的计算：
    # 在inner_loss计算中，用初始参数params首先根据observations得到actions，因为生成actions的policy是
    # “with a `Normal` distribution output, with trainable standard deviation”的，
    # 所以第二次将observations送入policy得到pi时，其第二次的动作与第一次的动作已经有所差别，而概率log_prob就是计算
    # 这两个动作之间的差别的，利用log_prob计算loss，根据初始参数params计算的；
    # 因此在计算surrogate_loss的时候，得到了updated的参数，这套参数也是根据validate_episode_observation计算
    # 了第二次的动作，用第二次动作与validate_episode_actions的差别计算概率log_prob，而计算loss，这个loss是
    # 根据adaptive params计算的。
    # （虽然不明白为什么要用detach_distribution抽离出old_pi）;
    def inner_loss(self, episodes, params=None):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        # Using learned weight to calculate returns (cumulative rewards) in baseline.
        values = self.baseline(episodes)
        # Suppose the shape of value is [2, 4, 1], the shape of advantages is [2, 4] (2 steps and 4 samples in a batch).
        # weighted_normalize advantages is also [2, 4].
        # Combine returns calculated by baseline and true rewards to get gae rewards.
        # GUESS: By using GAE, the rewards of baseline had been deducted from the advantages.
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)
        # episodes.observations: [2,4,3] -> pi: [2,4,5] (2:steps, 4:samples, 5:action_size)
        # Using policy's mlp to compute actions, i.e., pi, from observations.
        pi = self.policy(episodes.observations, params=params)
        # Using pi (computed by policy's mlp) and real actions to get log_probs of actions.
        # log_probs is like:
        # log_probs:torch.Size([2, 4, 5])
        # tensor([[[ -0.9389,  -0.9239,  -0.9639,  -2.0439,  -2.1989],
        #          [ -0.9189,  -1.0439,  -1.0989,  -1.5239,  -1.4189],
        #          [ -0.9189,  -0.9189,  -0.9639,  -7.7639,  -1.6389],
        #          [ -0.9189,  -0.9189,  -1.0439,  -0.9239, -94.7639]],
        #
        #         [[ -1.0989,  -0.9389,  -1.2389,  -5.4189,  -1.4189],
        #          [ -0.9989,  -1.0989,  -1.2389,  -1.4189, -50.9189],
        #          [ -0.9239,  -1.0989, -71.7239,  -0.9189,  -0.9189],
        #          [ -0.9389,  -0.9639,  -0.9189,  -0.9189,  -0.9239]]],
        #        grad_fn=<SubBackward>)
        # NOTE: pi is the predicted actions, why log_prod is combined with the predicted actions pi and another episodes.actions?
        # Answer: see notes before the self.sigma.data.fill_(math.log(init_std)).
        # GUESS: The probability here is that how possibly the value will be 'variant' to current value
        # based on normal distribution N(loc, scale).
        log_probs = pi.log_prob(episodes.actions)
        # log_probs are transformed into [2, 4]:
        # tensor([[ -7.0697,  -6.0047, -12.2047, -98.5697],
        #         [-10.1147, -55.6747, -75.5847,  -4.6647]], grad_fn=<SumBackward1>)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        # Multiply log_probs and advantages to get losses, which are used in REINFORCE.
        # advantages:
        # tensor([[-1.0911, -0.6547, -0.2182,  0.2182],
        #         [-1.5275,  0.6547,  1.0911,  1.5275]])
        # loss:
        # tensor(14.7243, grad_fn=<NegBackward>)
        # The loss is the average loss of all samples in one batch.
        loss = -weighted_mean(log_probs * advantages, dim=0,
            weights=episodes.mask)
        return loss

    def adapt(self, episodes, first_order=False):
        """Adapt the parameters of the policy network to a new task, from 
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Fit the baseline to the training episodes: set the nn.Linear.weight for linear layer used in baseline.
        # The linear layer in baseline is used to transform features into returns (cumulative rewards).
        self.baseline.fit(episodes)
        # Get the average loss (log_probability of actions * advantages of rewards) on the training episodes in one batch.
        loss = self.inner_loss(episodes)
        # Get the new parameters after a one-step gradient update
        # Each module parameter is computed as parameter = parameter - step_size * grad.
        # When being saved in the OrderedDict of self.named_parameters(), it likes:
        # OrderedDict([('sigma', Parameter containing:
        # tensor([0.6931, 0.6931], requires_grad=True)), ('0.weight', Parameter containing:
        # tensor([[1., 1.],
        #         [1., 1.]], requires_grad=True)), ('0.bias', Parameter containing:
        # tensor([0., 0.], requires_grad=True)), ('1.weight', Parameter containing:
        # tensor([[1., 1.],
        #         [1., 1.]], requires_grad=True)), ('1.bias', Parameter containing:
        # tensor([0., 0.], requires_grad=True))])
        params = self.policy.update_params(loss, step_size=self.fast_lr,
            first_order=first_order)

        return params

    def sample(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        Here number of tasks is 8.
        """
        episodes = []
        for task in tasks:
            # Reset task in current specific environment.
            # When reset_task, the env knows a new task is activated.
            self.sampler.reset_task(task)
            # Get all _observations_list, _actions_list and _rewards_list for all time steps
            # in fast-batch sized (20) episodes for a task (20-shot).
            # Since params is none, the parameters used in policy will be initialized as random values.
            train_episodes = self.sampler.sample(self.policy,
                gamma=self.gamma, device=self.device)
            # Get the average loss (log_probability of actions * advantages of rewards) on the training episodes in one batch.
            # Using the average loss to compute updated_parameter = parameter - step_size * grad.
            # Updated_parameter is saved in OrderedDict.
            params = self.adapt(train_episodes, first_order=first_order)
            # Get all _observations_list, _actions_list and _rewards_list in a new set of episodes for a batch by using multi-layer perceptron.
            # Using updated parameters to get valid_episodes.
            # The updated parameters are saved in params, but the initialized params in self.named_parameters() are not changed.
            valid_episodes = self.sampler.sample(self.policy, params=params,
                gamma=self.gamma, device=self.device)
            # For one task, train_episodes and valid_episodes are saved in a tuple.
            # All tasks will form a list:
            # [(train_episodes_0, valid_episodes_0),..., (train_episodes_(tasksize-1), valid_episodes_(tasksize-1))]
            episodes.append((train_episodes, valid_episodes))
        return episodes

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(),
                create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        losses, kls, pis = [], [], []
        if old_pis is None:
            # It is: [None, None, None,..., None, None, None, None, None], a list of 40 'None'.
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            # Self.policy.named_parameters() are initialized when the model is established.
            # The updated parameters are saved in params but the value of named_parameters() is not changed.
            # By using saved train_episodes in one task, the adapted parameters which are used in this task is generated.
            params = self.adapt(train_episodes)
            # with torch.set_grad_enabled (true or false): to set the operations in the block be able to compute gradient.
            # The torch.set_grad_enabled line of code makes sure to clear the intermediate values for evaluation,
            # which are needed to backpropagate during training, thus saving memory.
            # It’s comparable to the with torch.no_grad() statement but takes a bool value.
            # All new operations in the torch.set_grad_enabled(False) block won’t require gradients.
            # However, the model parameters will still require gradients.
            with torch.set_grad_enabled(old_pi is None):
                # episodes.observations: [2,4,3] -> pi: [2,4,5] (2:steps, 4:samples, 5:action_size)
                # Using policy's mlp to compute actions, i.e., pi, from observations.
                pi = self.policy(valid_episodes.observations, params=params)
                # detach_distribution(pi) = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
                # NOTE: Why detach_distribution(pi)?
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)
                # Using learned weight to calculate returns (cumulative rewards) in baseline.
                values = self.baseline(valid_episodes)
                # Get advantages of valid_episodes.
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages,
                    weights=valid_episodes.mask)
                # NOTE: WHY?
                # log_prob here is used to measure the variance of actions computed by `Normal` distribution output of NormalMLPPolicy。
                # Suppose an output action is a vector of elements,
                # then pi.log_prob(actions) is to compute the probability of each element in the action vector.
                # The probability of the action is computed as the product of probabilities of all elements in the action vector.
                # So the sum of elements in pi.log_prob(actions) is the log_prob of the whole action.
                log_ratio = (pi.log_prob(valid_episodes.actions)
                    - old_pi.log_prob(valid_episodes.actions))
                # [2, 4, 5] -> [2, 4]
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                # For example: tensor(-0.3125, grad_fn=<NegBackward>);
                loss = -weighted_mean(ratio * advantages, dim=0,
                    weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    # From:
                    # mask: torch.Size([2, 4])
                    # tensor([[1., 1., 1., 1.],
                    #         [0., 0., 0., 0.]])
                    # To:
                    # unsqueeze(2) mask: torch.Size([2, 4, 1])
                    # tensor([[[1.],
                    #          [1.],
                    #          [1.],
                    #          [1.]],
                    #
                    #         [[0.],
                    #          [0.],
                    #          [0.],
                    #          [0.]]])
                    mask = mask.unsqueeze(2)
                # For instance: kl: tensor(0., grad_fn=<MeanBackward1>)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0,
                    weights=mask)
                kls.append(kl)

        # torch.stack(tensors, dim=0, out=None) → Tensor:
        # Concatenates sequence of tensors along a new dimension into a list.
        # All tensors need to be of the same size.
        # Return the average of losses and kls of the batch of tasks,
        # and the pis of the batch of tasks as well.
        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), pis)

    def step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        # Get the average of losses of all episodes of all episodes in a batch of 40 tasks.
        # Get the detached distributions, old_pis for all episodes in a batch of 40 tasks.
        old_loss, _, old_pis = self.surrogate_loss(episodes)
        # torch.autograd.grad(outputs, inputs, grad_outputs=None,
        # retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False):
        # Computes and returns the sum of gradients of outputs w.r.t. the inputs.
        # outputs (sequence of Tensor) – outputs of the differentiated function.
        # inputs (sequence of Tensor) – Inputs w.r.t. which the gradient will be returned
        # (and not accumulated into .grad).
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        # Convert parameters to one vector, concat each parameter (vector) into one vector.
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes,
            damping=cg_damping)
        # hessian_vector_product here is the pointer of the function _product defined in hessian_vector_product
        # and will transferred into conjugate_gradient.
        stepdir = conjugate_gradient(hessian_vector_product, grads,
            cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            # Convert value of the vector into the value of parameters.
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
            # If new loss is less then old loss, which means having found better parameters.
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())
