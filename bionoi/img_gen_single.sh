colorby="residue_type"
mol_path=$1
output_folder=$2

mol=${mol_path##*/}
protein=${mol::${#mol}-5}

pop_path="${output_folder}/pop/${protein}.out"
profile_path="${output_folder}/profile/${protein}.profile"
output_folder="${output_folder}/output/"

python /home/tony/gits/BionoiNet/bionoi/bionoi.py -mol "$mol_path" -pop "$pop_path" -profile "$profile_path" -out "$output_folder" -colorby "$colorby" -direction 0 -rot_angle 0 -flip 0 