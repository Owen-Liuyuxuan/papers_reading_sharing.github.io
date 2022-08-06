commit_statement="add"
commit_statement=$commit_statement$1

echo "git pull"
git pull
echo "git add *"
git add *
echo "git commit"
git commit -m "$commit_statement"
echo "git push"
git push https://github.com/Owen-Liuyuxuan/papers_reading_sharing.github.io.git
echo "mkdocs gh-deploy"
mkdocs gh-deploy
echo "update local database"
python3 read_all_paper.py
