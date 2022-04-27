#!/usr/bin/env bash
set -e # halt script on error

export JEKYLL_ENV=production

bundle exec jekyll build
# Researchgate blocks requests from Travis
bundle exec htmlproofer \
    --internal-domains "borjafreire.github.io" \
    --url-ignore "https://www.researchgate.net/profile/Borja-Freire-Castro/" \
    ./_site
touch ./_site/.nojekyll