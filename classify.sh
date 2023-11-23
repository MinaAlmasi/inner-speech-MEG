source ./env/bin/activate

labels=(
    rh.bankssts.label
    lh.bankssts.label

    rh.medialorbitofrontal.label
    lh.medialorbitofrontal.label

    rh.superiortemporal.label
    lh.superiortemporal.label
)

for label in "${labels[@]}"; do
    echo "Running $label"
    python src/classify.py -label "$label"
done



deactivate