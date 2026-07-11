#!/usr/bin/env bash
# Fetch the small networks used in the Figure 1 reproduction (Karate is a
# networkx builtin and needs no download -- see src/polardrl/datasets.py).
#
# Dolphins and Netscience come from KONECT (http://konect.cc); Diseasome is
# NOT on KONECT under the moreno_disease slug the vault's dataset README
# originally guessed (verified 404 as of 2026-07-09) -- it's fetched from
# the Network Data Repository instead (networkrepository.com/bio-diseasome),
# which hosts the same Goh et al. Human Disease Network data (confirmed
# 516 nodes / 1188 edges, matching Zhu et al. 2021 Table 1 exactly).
#
# Idempotent: skips any network whose expected output file already exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/raw"
mkdir -p "${RAW_DIR}"

fetch_konect() {
  local name="$1" slug="$2"
  local target_dir="${RAW_DIR}/${name}"

  if compgen -G "${target_dir}/out.*" > /dev/null 2>&1; then
    echo "[skip] ${name}: ${target_dir}/out.* already present"
    return
  fi

  echo "[fetch] ${name} (KONECT slug: ${slug})"
  mkdir -p "${target_dir}"
  local tmp_archive
  tmp_archive="$(mktemp)"

  local url="http://konect.cc/files/download.tsv.${slug}.tar.bz2"
  if ! curl -fsSL "${url}" -o "${tmp_archive}"; then
    echo "[error] failed to download ${url}" >&2
    echo "        KONECT may be unreachable or the slug may have changed --" >&2
    echo "        check http://konect.cc/networks/${slug}/ manually." >&2
    rm -f "${tmp_archive}"
    exit 1
  fi

  tar -xjf "${tmp_archive}" -C "${target_dir}" --strip-components=1
  rm -f "${tmp_archive}"

  if ! compgen -G "${target_dir}/out.*" > /dev/null 2>&1; then
    echo "[error] extracted ${name} but found no out.* edge list in ${target_dir}" >&2
    exit 1
  fi
  echo "[done] ${name} -> ${target_dir}"
}

fetch_diseasome() {
  local target_dir="${RAW_DIR}/diseasome"

  if compgen -G "${target_dir}/*.mtx" > /dev/null 2>&1; then
    echo "[skip] diseasome: ${target_dir}/*.mtx already present"
    return
  fi

  echo "[fetch] diseasome (Network Data Repository: bio-diseasome)"
  mkdir -p "${target_dir}"
  local tmp_zip
  tmp_zip="$(mktemp --suffix=.zip)"

  local url="https://nrvis.com/download/data/bio/bio-diseasome.zip"
  if ! curl -fsSL "${url}" -o "${tmp_zip}"; then
    echo "[error] failed to download ${url}" >&2
    echo "        check https://networkrepository.com/bio-diseasome.php manually." >&2
    rm -f "${tmp_zip}"
    exit 1
  fi

  unzip -oq "${tmp_zip}" -d "${target_dir}"
  rm -f "${tmp_zip}"

  if ! compgen -G "${target_dir}/*.mtx" > /dev/null 2>&1; then
    echo "[error] extracted diseasome but found no .mtx file in ${target_dir}" >&2
    exit 1
  fi
  echo "[done] diseasome -> ${target_dir}"
}

fetch_snap() {
  local name="$1" slug="$2"
  local target_dir="${RAW_DIR}/${name}"
  local target_file="${target_dir}/${slug}.txt"

  if [[ -f "${target_file}" ]]; then
    echo "[skip] ${name}: ${target_file} already present"
    return
  fi

  echo "[fetch] ${name} (SNAP: ${slug})"
  mkdir -p "${target_dir}"

  local url="https://snap.stanford.edu/data/${slug}.txt.gz"
  if ! curl -fsSL "${url}" -o "${target_file}.gz"; then
    echo "[error] failed to download ${url}" >&2
    echo "        check https://snap.stanford.edu/data/${slug}.html manually." >&2
    rm -f "${target_file}.gz"
    exit 1
  fi

  gunzip "${target_file}.gz"
  echo "[done] ${name} -> ${target_file}"
}

fetch_konect dolphins dolphins
fetch_konect netscience dimacs10-netscience
fetch_diseasome
# GrQc: smallest network in Table 1 -- the paper reports SPGREEDY's exact
# Delta I(G) = -4.9966 at k=50 for this one, giving us a real number to
# match instead of an eyeballed Figure 1 curve overlap.
fetch_snap grqc ca-GrQc

echo "All datasets present. Karate needs no download (networkx builtin)."
