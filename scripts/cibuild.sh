#!/usr/bin/env bash
set -e # halt script on error

export JEKYLL_ENV=production

bundle exec jekyll build
# Researchgate blocks requests from Travis
bundle exec htmlproofer \
    --internal-domains "chponte.github.io" \
    --url-ignore "https://www.researchgate.net/profile/Christian_Ponte-Fernandez" \
    ./_site
touch ./_site/.nojekyll