for repo in \
  "https://github.com/apache/commons-bcel resources/datasets/commons-bcel" \
  "https://github.com/apache/commons-beanutils resources/datasets/commons-beanutils" \
  "https://github.com/apache/commons-bsf resources/datasets/commons-bsf" \
  "https://github.com/apache/commons-cli resources/datasets/commons-cli" \
  "https://github.com/apache/commons-codec resources/datasets/commons-codec" \
  "https://github.com/apache/commons-collections resources/datasets/commons-collections" \
  "https://github.com/apache/commons-configuration resources/datasets/commons-configuration" \
  "https://github.com/apache/commons-crypto resources/datasets/commons-crypto" \
  "https://github.com/apache/commons-csv resources/datasets/commons-csv" \
  "https://github.com/apache/commons-dbcp resources/datasets/commons-dbcp" \
  "https://github.com/apache/commons-dbutils resources/datasets/commons-dbutils" \
  "https://github.com/apache/commons-email resources/datasets/commons-email" \
  "https://github.com/apache/commons-exec resources/datasets/commons-exec" \
  "https://github.com/apache/commons-fileupload resources/datasets/commons-fileupload" \
  "https://github.com/apache/commons-imaging resources/datasets/commons-imaging" \
  "https://github.com/apache/commons-io resources/datasets/commons-io" \
  "https://github.com/apache/commons-jcs resources/datasets/commons-jcs" \
  "https://github.com/apache/commons-jexl resources/datasets/commons-jexl" \
  "https://github.com/apache/commons-lang resources/datasets/commons-lang" \
  "https://github.com/apache/commons-logging resources/datasets/commons-logging" \
  "https://github.com/apache/commons-math resources/datasets/commons-math" \
  "https://github.com/apache/commons-net resources/datasets/commons-net" \
  "https://github.com/apache/commons-numbers resources/datasets/commons-numbers" \
  "https://github.com/apache/commons-pool resources/datasets/commons-pool" \
  "https://github.com/apache/commons-text resources/datasets/commons-text" \
  "https://github.com/apache/commons-validator resources/datasets/commons-validator" \
  "https://github.com/apache/commons-vfs resources/datasets/commons-vfs"
do
  set -- $repo
  git submodule add -f "$1" "$2"
done
