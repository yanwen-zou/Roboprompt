from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class NoopImageRunner(BaseImageRunner):
    def run(self, policy):
        del policy
        return {}
