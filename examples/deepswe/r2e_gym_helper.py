# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper utilities and runtime monkeypatches for DeepSWE and R2E-Gym on GKE."""

import logging
import os


def patch_agent_sandbox_resources():
  """Strip resource requests from agent_sandbox_rl templates to prevent Kueue throttling on GKE."""
  for path in [
      "/opt/venv/lib/python3.12/site-packages/agent_sandbox_rl/resources.py",
  ]:
    if os.path.exists(path):
      try:
        with open(path) as f:
          content = f.read()
        target = (
            '"resources": {"requests": {\n'
            '                "cpu": template.resources.cpu,\n'
            '                "memory": template.resources.memory,\n'
            '            }},'
        )
        if target in content:
          with open(path, "w") as f:
            f.write(content.replace(target, '"resources": {},'))
          logging.info("[Monkeypatch] Patched agent_sandbox_rl resources in %s", path)
      except Exception as e:
        logging.warning("[Monkeypatch] Could not patch %s: %s", path, e)


def patch_kubernetes_api_client():
  """Ensure kubernetes.client.api_client handles non-bytes exception bodies cleanly on Python 3.12."""
  for path in [
      "/opt/venv/lib/python3.12/site-packages/kubernetes/client/api_client.py",
  ]:
    if os.path.exists(path):
      try:
        with open(path) as f:
          content = f.read()
        target = "e.body = e.body.decode('utf-8') if six.PY3 else e.body"
        replacement = (
            "e.body = (e.body.decode('utf-8') if hasattr(e.body, 'decode') "
            "else str(e.body)) if six.PY3 else e.body"
        )
        if target in content:
          with open(path, "w") as f:
            f.write(content.replace(target, replacement))
          logging.info("[Monkeypatch] Patched kubernetes api_client in %s", path)
      except Exception as e:
        logging.warning("[Monkeypatch] Could not patch %s: %s", path, e)


def patch_k8s_watch_stream():
  """Wrap K8sHelper._watch_claim to transparently retry on premature HTTP stream disconnections."""
  try:
    import http.client
    from k8s_agent_sandbox.k8s_helper import K8sHelper
    import urllib3.exceptions

    orig_watch_claim = K8sHelper._watch_claim

    def robust_watch_claim(
        self,
        claim_name: str,
        namespace: str,
        timeout: int,
        require_ready: bool,
        resource_version: str | None = None,
    ) -> str:
      import time
      deadline = time.monotonic() + timeout
      rv = resource_version or "0"
      while True:
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
          raise TimeoutError(
              f"Could not resolve claim '{claim_name}' within {timeout} seconds."
          )
        try:
          return orig_watch_claim(
              self,
              claim_name,
              namespace,
              remaining,
              require_ready,
              resource_version=rv,
          )
        except (
            urllib3.exceptions.HTTPError,
            http.client.HTTPException,
            ConnectionError,
        ) as net_err:
          logging.warning(
              "[K8sWatchRetry] Watch on claim '%s' disconnected (%s: %s); retrying with rv=0",
              claim_name,
              type(net_err).__name__,
              net_err,
          )
          rv = "0"
          time.sleep(0.5)

    K8sHelper._watch_claim = robust_watch_claim
    logging.info("[Monkeypatch] Successfully wrapped K8sHelper._watch_claim with robust HTTP retry")
  except Exception as e:
    logging.warning("[Monkeypatch] Failed to patch K8sHelper._watch_claim: %s", e)


def patch_kubernetes_runtime():
  """Monkeypatch r2egym DockerRuntime to dynamically configure Kubernetes nodeSelector.

  This is required because r2egym hardcodes the CPU nodepool name (using
  Karpenter bigcpu-standby), which does not exist in GKE clusters. We
  override it to match the nodepool configured via NODE_SELECTOR_KEY and
  NODE_SELECTOR_VAL environment variables.
  """
  patch_agent_sandbox_resources()
  patch_kubernetes_api_client()
  patch_k8s_watch_stream()
  try:
    from r2egym.agenthub.runtime.docker import DockerRuntime

    original_start_kubernetes_pod = DockerRuntime._start_kubernetes_pod

    def patched_start_kubernetes_pod(
        self, docker_image, command, pod_name, **docker_kwargs
    ):
      original_create_namespaced_pod = self.client.create_namespaced_pod

      def patched_create_namespaced_pod(*args, **kwargs):
        body = kwargs.get("body")
        if body and "spec" in body:
          key = os.environ.get(
              "NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool"
          )
          val = os.environ.get("NODE_SELECTOR_VAL", "cpu-np")
          body["spec"]["nodeSelector"] = {key: val}
          logging.info("[Monkeypatch] Overrode nodeSelector to %s=%s", key, val)
        return original_create_namespaced_pod(*args, **kwargs)

      self.client.create_namespaced_pod = patched_create_namespaced_pod
      try:
        return original_start_kubernetes_pod(
            self, docker_image, command, pod_name, **docker_kwargs
        )
      finally:
        self.client.create_namespaced_pod = original_create_namespaced_pod

    DockerRuntime._start_kubernetes_pod = patched_start_kubernetes_pod
    logging.info(
        "[Monkeypatch] Successfully patched DockerRuntime._start_kubernetes_pod"
    )
  except Exception as e:
    logging.warning("[Monkeypatch] Failed to patch DockerRuntime: %s", e)
