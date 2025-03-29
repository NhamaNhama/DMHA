from kubernetes import client, config

class ClusterManager:
    def __init__(self):
        config.load_incluster_config()
        self.api = client.CoreV1Api()
        self.apps_api = client.AppsV1Api()

    def scale_deployment(self, deployment_name: str, replicas: int, namespace: str = "default"):
        body = {
            "spec": {
                "replicas": replicas
            }
        }
        self.apps_api.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body=body
        )