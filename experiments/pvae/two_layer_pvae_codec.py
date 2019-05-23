import numpy as np

from craystack.bb_ans import BBANS
from craystack import codecs


def AutoRegressive_return_params(elem_param_fn, data_shape, params_shape, elem_idxs, elem_codec):
    def append(message, data, all_params=None):
        if all_params is None:
            all_params = np.zeros(params_shape, dtype=np.float32)  # ?
            all_params = elem_param_fn(data, all_params)
        for idx in reversed(elem_idxs):
            elem_params = all_params[idx]
            elem_append, _ = elem_codec(elem_params, idx)
            message = elem_append(message, data[idx].astype('uint64'))
        return message

    def pop(message):
        data = np.zeros(data_shape, dtype=np.uint64)
        all_params = np.zeros(params_shape, dtype=np.float32)
        for idx in elem_idxs:
            all_params = elem_param_fn(data, all_params, idx)
            elem_params = all_params[idx]
            _, elem_pop = elem_codec(elem_params, idx)
            message, elem = elem_pop(message)
            data[idx] = elem
        return message, (data, all_params)
    return append, pop


def TwoLayerVAE(gen_net2_partial,
                rec_net1, rec_net2,
                post1_codec, obs_codec,
                prior_prec, latent_prec,
                get_theta):
    """
    rec_net1 outputs params for q(z1|x)
    rec_net2 outputs params for q(z2|x)
    post1_codec is to code z1 by q(z1|z2,x)
    obs_codec is to code x by p(x|z1)"""
    z1_view = lambda head: head[0]
    z2_view = lambda head: head[1]
    x_view = lambda head: head[2]

    prior_z1_append, prior_z1_pop = codecs.substack(Uniform(prior_prec), z1_view)
    prior_z2_append, prior_z2_pop = codecs.substack(Uniform(prior_prec), z2_view)

    def prior_append(message, latent):
        (z1, z2), theta1 = latent
        message = prior_z1_append(message, z1)
        message = prior_z2_append(message, z2)
        return message

    def prior_pop(message):
        message, z2 = prior_z2_pop(message)
        message, z1 = prior_z1_pop(message)
        # compute theta1
        eps1_vals = codecs.std_gaussian_centres(prior_prec)[z1]
        z2_vals = codecs.std_gaussian_centres(prior_prec)[z2]
        theta1 = get_theta(eps1_vals, z2_vals)
        return message, ((z1, z2), theta1)

    def likelihood(latent):
        (z1, _), theta1 = latent
        # get z1_vals from the latent
        _, _, mu1_prior, sig1_prior = np.moveaxis(theta1, -1, 0)
        eps1_vals = codecs.std_gaussian_centres(prior_prec)[z1]
        z1_vals = mu1_prior + sig1_prior * eps1_vals
        append, pop = codecs.substack(obs_codec(gen_net2_partial(z1_vals)), x_view)
        return append, pop

    def posterior(data):
        mu1, sig1, h = rec_net1(data)
        mu2, sig2 = rec_net2(h)

        post_z2_append, post_z2_pop = codecs.substack(DiagGaussian_StdBins(
            mu2, sig2, latent_prec, prior_prec), z2_view)

        def posterior_append(message, latents):
            (z1, z2), theta1 = latents
            z2_vals = codecs.std_gaussian_centres(prior_prec)[z2]
            post_z1_append, _ = codecs.substack(post1_codec(z2_vals, mu1, sig1), z1_view)
            theta1[..., 0] = mu1
            theta1[..., 1] = sig1

            message = post_z1_append(message, z1, theta1)
            message = post_z2_append(message, z2)
            return message

        def posterior_pop(message):
            message, z2 = post_z2_pop(message)
            z2_vals = codecs.std_gaussian_centres(prior_prec)[z2]
            # need to return theta1 from the z1 pop
            _, post_z1_pop = codecs.substack(post1_codec(z2_vals, mu1, sig1), z1_view)
            message, (z1, theta1) = post_z1_pop(message)
            return message, ((z1, z2), theta1)

        return posterior_append, posterior_pop

    return BBANS((prior_append, prior_pop), likelihood, posterior)
