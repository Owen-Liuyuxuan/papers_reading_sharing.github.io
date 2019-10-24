commit_statement="add "
commit_statement=$commit_statement$1


git add *
git commit -m "$commit_statement"
git push https://github.com/Owen-Liuyuxuan/papers_reading_sharing.github.io.git

mkdocs gh-deploy
