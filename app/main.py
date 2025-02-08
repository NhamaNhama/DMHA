import uvicorn
from .config import Config
from .context_system import ContextAwareSystem
from .api_service import APIService

def create_app():
    config_obj = Config()
    system = ContextAwareSystem(config_obj)
    system.setup_k8s_cluster_manager()
    api_service = APIService(system)
    return api_service.app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        timeout_keep_alive=30
    )