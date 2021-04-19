import numpy as np


def rejection_sample(
    n, sample_fn, logpdf_sampling, logpdf_target, logmaxratio, maxiter: int = 100
):
    def total_len(ss):
        return sum(len(s) for s in ss)

    samples = []

    counter = 0
    while total_len(samples) < n:
        samples_proposed = sample_fn(n)
        p = logpdf_sampling(samples_proposed)
        q = logpdf_target(samples_proposed)
        u = np.log(np.random.rand(*p.shape))

        indkeep = u <= q - logmaxratio - p
        samples_kept = np.reshape(
            samples_proposed[indkeep, ...], (-1, *samples_proposed.shape[1:])
        )
        samples.append(samples_kept)
        counter += 1

        if counter > maxiter:
            print("Reached maxiter.")
            break
    return np.concatenate(samples, axis=0)[:n]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.stats

    # Example use
    target = scipy.stats.norm(loc=0.5, scale=0.1)
    sampling = scipy.stats.uniform()

    thetas = np.linspace(0, 1, 1000)
    target_logpdfs = target.logpdf(thetas)
    sampling_logpdfs = sampling.logpdf(thetas)
    logmaxratio = max(target_logpdfs) - max(sampling_logpdfs)

    out = rejection_sample(
        10000, sampling.rvs, sampling.logpdf, target.logpdf, logmaxratio
    )
    _ = plt.hist(out, bins=100)
