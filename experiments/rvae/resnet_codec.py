from craystack.bb_ans import BBANS
from craystack import codecs

def ResNetVAE(up_pass, rec_net_top, rec_nets, gen_net_top, gen_nets, obs_codec,
              prior_prec, latent_prec):
    """
    Codec for a ResNetVAE.
    Assume that the posterior is bidirectional -
    i.e. has a deterministic upper pass but top down sampling.
    Further assume that all latent conditionals are factorised Gaussians,
    both in the generative network p(z_n|z_{n-1})
    and in the inference network q(z_n|x, z_{n-1})

    Assume that everything is ordered bottom up
    """
    z_view = lambda head: head[0]
    x_view = lambda head: head[1]

    prior_codec = codecs.substack(codecs.Uniform(prior_prec), z_view)

    def prior_append(message, latents):
        # append bottom-up
        append, _ = prior_codec
        latents, _ = latents
        for latent in latents:
            latent, _ = latent
            message = append(message, latent)
        return message

    def prior_pop(message):
        # pop top-down
        (prior_mean, prior_stdd), h_gen = gen_net_top()
        _, pop = prior_codec
        message, latent = pop(message)
        latents = [(latent, (prior_mean, prior_stdd))]
        for gen_net in reversed(gen_nets):
            previous_latent_val = prior_mean + codecs.std_gaussian_centres(prior_prec)[latent] * prior_stdd
            (prior_mean, prior_stdd), h_gen = gen_net(h_gen, previous_latent_val)
            message, latent = pop(message)
            latents.append((latent, (prior_mean, prior_stdd)))
        return message, (latents[::-1], h_gen)

    def posterior(data):
        # run deterministic upper-pass
        contexts = up_pass(data)

        def posterior_append(message, latents):
            # first run the model top-down to get the params and latent vals
            latents, _ = latents

            (post_mean, post_stdd), h_rec = rec_net_top(contexts[-1])
            post_params = [(post_mean, post_stdd)]

            for rec_net, latent, context in reversed(list(zip(rec_nets, latents[1:], contexts[:-1]))):
                previous_latent, (prior_mean, prior_stdd) = latent
                previous_latent_val = prior_mean + \
                                      codecs.std_gaussian_centres(prior_prec)[previous_latent] * prior_stdd

                (post_mean, post_stdd), h_rec = rec_net(h_rec, previous_latent_val, context)
                post_params.append((post_mean, post_stdd))

            # now append bottom up
            for latent, post_param in zip(latents, reversed(post_params)):
                latent, (prior_mean, prior_stdd) = latent
                post_mean, post_stdd = post_param
                append, _ = codecs.substack(codecs.DiagGaussian_GaussianBins(post_mean, post_stdd,
                                                           prior_mean, prior_stdd,
                                                           latent_prec, prior_prec),
                                        z_view)
                message = append(message, latent)
            return message

        def posterior_pop(message):
            # pop top-down
            (post_mean, post_stdd), h_rec = rec_net_top(contexts[-1])
            (prior_mean, prior_stdd), h_gen = gen_net_top()
            _, pop = codecs.substack(codecs.DiagGaussian_GaussianBins(post_mean, post_stdd,
                                                    prior_mean, prior_stdd,
                                                    latent_prec, prior_prec),
                                 z_view)
            message, latent = pop(message)
            latents = [(latent, (prior_mean, prior_stdd))]
            for rec_net, gen_net, context in reversed(list(zip(rec_nets, gen_nets, contexts[:-1]))):
                previous_latent_val = prior_mean + \
                                      codecs.std_gaussian_centres(prior_prec)[latents[-1][0]] * prior_stdd

                (post_mean, post_stdd), h_rec = rec_net(h_rec, previous_latent_val, context)
                (prior_mean, prior_stdd), h_gen = gen_net(h_gen, previous_latent_val)
                _, pop = codecs.substack(codecs.DiagGaussian_GaussianBins(post_mean, post_stdd,
                                                        prior_mean, prior_stdd,
                                                        latent_prec, prior_prec),
                                     z_view)
                message, latent = pop(message)
                latents.append((latent, (prior_mean, prior_stdd)))
            return message, (latents[::-1], h_gen)

        return posterior_append, posterior_pop

    def likelihood(latents):
        # get the z1 vals to condition on
        latents, h = latents
        z1_idxs, (prior_mean, prior_stdd) = latents[0]
        z1_vals = prior_mean + codecs.std_gaussian_centres(prior_prec)[z1_idxs] * prior_stdd
        return codecs.substack(obs_codec(h, z1_vals), x_view)

    return BBANS((prior_append, prior_pop), likelihood, posterior)
