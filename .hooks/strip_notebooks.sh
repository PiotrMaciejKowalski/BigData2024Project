#!/usr/bin/env bash -e

IFS=$'\n'

if [  "$GITLAB_CI" == true ]; then
    files=($(git ls-files))
else
    files=($(git diff --cached --name-only --diff-filter=ACM))
    should_git_add=true
fi

for file in "${files[@]}" ; do
    if [[ $file == *.ipynb ]] ;
    then
        nb_dir=$(dirname "$file")
        if [[ $nb_dir == "." ]]; then
            nb_dir=""
        fi

        filename=$(basename "$file")
        stripped_dir=stripped/${nb_dir} 
        mkdir -p "$stripped_dir"
        target_stripped_file="${stripped_dir}/${filename%.ipynb}_stripped.py"

        jupyter nbconvert --to script $file
        echo -e "Filename $filename"
        echo -e "Filename ${filename%.ipynb}"
        mv ${file%.ipynb}.txt $target_stripped_file
		
        if [  "$should_git_add" == true ]; then 
          git add "$target_stripped_file" 
        fi

    fi
done
