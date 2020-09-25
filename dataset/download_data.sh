cd "$(dirname "$0")"

aws s3 cp s3://mint-solid-mnist/bin_test_clean.7z ./
aws s3 cp s3://mint-solid-mnist/bin_train.7z ./

7z x bin_test_clean.7z -obin
7z x bin_train.7z -obin
