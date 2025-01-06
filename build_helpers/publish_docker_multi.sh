#!/bin/sh

# The below assumes a correctly setup docker buildx environment

IMAGE_NAME=traderpilotorg/traderpilot
CACHE_IMAGE=traderpilotorg/traderpilot_cache
# Replace / with _ to create a valid tag
TAG=$(echo "${BRANCH_NAME}" | sed -e "s/\//_/g")
TAG_PLOT=${TAG}_plot
TAG_TRADERAI=${TAG}_traderai
TAG_TRADERAI_RL=${TAG_TRADERAI}rl
TAG_PI="${TAG}_pi"

PI_PLATFORM="linux/arm/v7"
echo "Running for ${TAG}"
CACHE_TAG=${CACHE_IMAGE}:${TAG_PI}_cache

# Add commit and commit_message to docker container
echo "${GITHUB_SHA}" > traderpilot_commit

if [ "${GITHUB_EVENT_NAME}" = "schedule" ]; then
    echo "event ${GITHUB_EVENT_NAME}: full rebuild - skipping cache"
    # Build regular image
    docker build -t traderpilot:${TAG} .
    # Build PI image
    docker buildx build \
        --cache-to=type=registry,ref=${CACHE_TAG} \
        -f docker/Dockerfile.armhf \
        --platform ${PI_PLATFORM} \
        -t ${IMAGE_NAME}:${TAG_PI} \
        --push \
        --provenance=false \
        .
else
    echo "event ${GITHUB_EVENT_NAME}: building with cache"
    # Build regular image
    docker pull ${IMAGE_NAME}:${TAG}
    docker build --cache-from ${IMAGE_NAME}:${TAG} -t traderpilot:${TAG} .

    # Pull last build to avoid rebuilding the whole image
    # docker pull --platform ${PI_PLATFORM} ${IMAGE_NAME}:${TAG}
    # disable provenance due to https://github.com/docker/buildx/issues/1509
    docker buildx build \
        --cache-from=type=registry,ref=${CACHE_TAG} \
        --cache-to=type=registry,ref=${CACHE_TAG} \
        -f docker/Dockerfile.armhf \
        --platform ${PI_PLATFORM} \
        -t ${IMAGE_NAME}:${TAG_PI} \
        --push \
        --provenance=false \
        .
fi

if [ $? -ne 0 ]; then
    echo "failed building multiarch images"
    return 1
fi
# Tag image for upload and next build step
docker tag traderpilot:$TAG ${CACHE_IMAGE}:$TAG

docker build --build-arg sourceimage=traderpilot --build-arg sourcetag=${TAG} -t traderpilot:${TAG_PLOT} -f docker/Dockerfile.plot .
docker build --build-arg sourceimage=traderpilot --build-arg sourcetag=${TAG} -t traderpilot:${TAG_TRADERAI} -f docker/Dockerfile.traderai .
docker build --build-arg sourceimage=traderpilot --build-arg sourcetag=${TAG_TRADERAI} -t traderpilot:${TAG_TRADERAI_RL} -f docker/Dockerfile.traderai_rl .

docker tag traderpilot:$TAG_PLOT ${CACHE_IMAGE}:$TAG_PLOT
docker tag traderpilot:$TAG_TRADERAI ${CACHE_IMAGE}:$TAG_TRADERAI
docker tag traderpilot:$TAG_TRADERAI_RL ${CACHE_IMAGE}:$TAG_TRADERAI_RL

# Run backtest
docker run --rm -v $(pwd)/tests/testdata/config.tests.json:/traderpilot/config.json:ro -v $(pwd)/tests:/tests traderpilot:${TAG} backtesting --datadir /tests/testdata --strategy-path /tests/strategy/strats/ --strategy StrategyTestV3

if [ $? -ne 0 ]; then
    echo "failed running backtest"
    return 1
fi

docker images

docker push ${CACHE_IMAGE}:$TAG
docker push ${CACHE_IMAGE}:$TAG_PLOT
docker push ${CACHE_IMAGE}:$TAG_TRADERAI
docker push ${CACHE_IMAGE}:$TAG_TRADERAI_RL

docker images

if [ $? -ne 0 ]; then
    echo "failed building image"
    return 1
fi
