declare -A short2long

short2long=(
    ["ar"]="arb_Arab"
    ["zh"]="zho_Hans"
    ["fi"]="fin_Latn"
    ["fr"]="fra_Latn"
    ["de"]="deu_Latn"
    ["ja"]="jpn_Jpan"
    ["it"]="ita_Latn"
    ["ko"]="kor_Hang"
    ["pt"]="por_Latn"
    ["ru"]="rus_Cyrl"
    ["es"]="spa_Latn"
    ["th"]="tha_Thai"
    ["en"]="eng_Latn"
)

cd ../..
path=$1
for folder in $(ls $path); do
    langshort=${folder: -2}
    langlong=${short2long[$langshort]}
    echo python3 eval.py --folder $path/$folder --lid $langlong
done