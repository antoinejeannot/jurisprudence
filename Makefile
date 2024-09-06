
.PHONY: export install compress upload release
include .env

# Define the date format for CalVer
YEAR := $(shell date +"%Y")
MONTH := $(shell date +"%m")
DAY := $(shell date +"%d")
VERSION := v$(YEAR).$(MONTH).$(DAY)

export:
	@mkdir -p raws
	@python jurisprudence.py export ./raws 

install:
	@pip install -r requirements.txt

compress:
	@mkdir -p compressed
	@find ./raws/CA -name "*.jsonl" -type f -print0 | xargs -0 tar czvf compressed/chambre_d_appel.jsonl.tar.gz -C . --files-from=-
	@find ./raws/TJ -name "*.jsonl" -type f -print0 | xargs -0 tar czvf compressed/tribunal_judiciaire.jsonl.tar.gz -C . --files-from=-
	@find ./raws/CC -name "*.jsonl" -type f -print0 | xargs -0 tar czvf compressed/cours_de_cassation.jsonl.tar.gz -C . --files-from=-

upload:
	@cp ./metadata.yaml > ./compressed/README.md
	@cat ./release_notes/$(VERSION).md >> ./compressed/README.md
	@huggingface-cli upload --repo-type=dataset --commit-message="✨ $(VERSION) 🏛️" --revision=main --include="*.jsonl.tar.gz" --include="*.md" ajeannot/jurisprudence ./compressed --quiet
	
release-note:
	@python jurisprudence.py release-note ./raws ./release_notes/$(VERSION).md
