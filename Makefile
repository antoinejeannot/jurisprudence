
.PHONY: export install compress upload release-note

# Define the date format for CalVer
YEAR := $(shell date +"%Y")
MONTH := $(shell date +"%m")
DAY := $(shell date +"%d")
VERSION := v$(YEAR).$(MONTH).$(DAY)

export:
	@mkdir -p raws
	@jurisprudence export ./raws --start-date 1860-01-01 -j CC
	@jurisprudence export ./raws --start-date 1996-01-01 -j CA
	@jurisprudence export ./raws --start-date 2020-01-01 -j TJ

install:
	@pip install -e ".[cli]"

compress:
	@mkdir -p compressed

	@find ./raws/CA -name "*.jsonl" -type f -print0 | sort -z | xargs -0 cat > compressed/cour_d_appel.jsonl 
	@rm -rf ./raws/CA
	@jurisprudence to-parquet compressed/cour_d_appel.jsonl
	@gzip compressed/cour_d_appel.jsonl

	@find ./raws/TJ -name "*.jsonl" -type f -print0 | sort -z | xargs -0 cat > compressed/tribunal_judiciaire.jsonl 
	@rm	-rf ./raws/TJ
	@jurisprudence to-parquet compressed/tribunal_judiciaire.jsonl
	@gzip compressed/tribunal_judiciaire.jsonl

	@find ./raws/CC -name "*.jsonl" -type f -print0 | sort -z | xargs -0 cat > compressed/cour_de_cassation.jsonl
	@rm	-rf ./raws/CC
	@jurisprudence to-parquet compressed/cour_de_cassation.jsonl
	@gzip compressed/cour_de_cassation.jsonl
	
release-note:
	@jurisprudence release-note ./compressed ./release_notes --version $(VERSION)
	@cp release_notes/$(VERSION).md README.md

upload:
	@cp ./metadata.md ./compressed/README.md
	@cat ./release_notes/$(VERSION).md >> ./compressed/README.md
	@huggingface-cli upload --repo-type=dataset --commit-message="✨ $(VERSION) 🏛️" --revision=main --include="*" antoinejeannot/jurisprudence ./compressed
