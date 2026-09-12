.PHONY: help install up down logs shell test test-embedded test-integration df prune reclaim nuke

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package plus dev, embedded and llm extras
	pip install -e '.[dev,embedded,llm]'

up:  ## Start Neo4j and wait for it to be healthy
	docker compose up -d --wait

down:  ## Stop Neo4j (data in ./neo4j survives)
	docker compose down

logs:  ## Tail Neo4j logs
	docker compose logs -f neo4j

shell:  ## Open a cypher-shell against the running container
	docker compose exec neo4j cypher-shell -u $${NEO4J_USER:-neo4j} -p $${NEO4J_PASSWORD:-password}

test:  ## Run all tests (Neo4j tests skip if no server is reachable)
	pytest -v

test-embedded:  ## Run the suite against the embedded backend only (no Docker needed)
	pytest -v -k embedded

test-integration:  ## Run the suite against Neo4j (requires `make up`)
	pytest -v -k neo4j

# ---------------------------------------------------------------------------
# Disk hygiene. On macOS the Docker VM lives in one sparse file that grows to a
# high-water mark and never shrinks by itself -- `prune` frees space *inside*
# the VM, `reclaim` is what actually returns it to macOS.
# ---------------------------------------------------------------------------

df:  ## Show what Docker is actually using
	docker system df -v

prune:  ## Remove unused images and build cache. Never touches ./neo4j.
	docker image prune -a -f
	docker builder prune -f

reclaim:  ## TRIM the VM disk so macOS gets the freed space back
	docker run --rm --privileged --pid=host docker/desktop-reclaim-space

nuke:  ## Delete the container AND the local graph data. Destructive.
	docker compose down -v
	rm -rf ./neo4j
