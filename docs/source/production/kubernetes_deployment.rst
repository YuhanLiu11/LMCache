Kubernetes deployment
=====================

To deploy LMCache in a Kubernetes cluster, you can use the `vLLM Production Stack <https://github.com/vllm-project/production-stack>`_.

Check out the documentation for the `vLLM Production Stack <https://docs.vllm.ai/projects/production-stack/en/latest/>`_ for more details.

A simple deployment example is shown below.

.. code-block:: bash

    helm repo add vllm https://vllm-project.github.io/production-stack
    helm install vllm vllm/vllm-stack -f tutorials/assets/values-06-shared-storage.yaml

.. The ``values-01-minimal-example.yaml`` file is located in the `production-stack/tutorials/assets <https://github.com/vllm-project/production-stack/tree/main/tutorials/assets>`_ directory.
The ``values-06-shared-storage.yaml`` file is shown below, and can be found in the `production-stack/tutorials/assets <https://github.com/vllm-project/production-stack/tree/main/tutorials/assets>`_ directory too.

.. code-block:: yaml

    servingEngineSpec:
        runtimeClassName: ""
        modelSpec:
        - name: "mistral"
            repository: "lmcache/vllm-openai"
            tag: "2025-04-18"
            modelURL: "mistralai/Mistral-7B-Instruct-v0.2"
            replicaCount: 2
            requestCPU: 10
            requestMemory: "40Gi"
            requestGPU: 1
            pvcStorage: "50Gi"
            vllmConfig:
            enableChunkedPrefill: false
            enablePrefixCaching: false
            maxModelLen: 16384
            v1: 0

            lmcacheConfig:
            enabled: true
            cpuOffloadingBufferSize: "20"
            env:
            - name: LMCACHE_LOG_LEVEL
                value: "DEBUG"
            hf_token: <YOUR HF TOKEN>

    cacheserverSpec:
    # -- Number of replicas
        replicaCount: 1

        # -- Container port
        containerPort: 8080

        # -- Service port
        servicePort: 81

        # -- Serializer/Deserializer type
        serde: "naive"

        # -- Cache server image (reusing the vllm image)
        repository: "lmcache/vllm-openai"
        tag: "2025-04-18"

        # TODO (Jiayi): please adjust this once we have evictor
        # -- router resource requests and limits
        resources:
            requests:
            cpu: "4"
            memory: "8G"
            limits:
            cpu: "4"
            memory: "10G"

        # -- Customized labels for the cache server deployment
        labels:
            environment: "cacheserver"
            release: "cacheserver"

In this configuration, the ``vllmConfig`` section is used to configure the vLLM engine, and the ``lmcacheConfig`` section is used to configure the LMCache. 
This file uses CPU offloading with LMCache and a remote LMCache server by setting cacheserverSpec.

The above example runs vLLM v0, but if you want to use vLLM v1, you can set the ``v1`` field to ``1`` in the ``vllmConfig`` section.

.. code-block:: yaml

    vllmConfig:
        v1: 1
