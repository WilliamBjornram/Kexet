from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
from open_spiel.python import games
import pyspiel
import pickle 

def export_tabular_policy(pickle_file_path: str, output_txt_path: str):
    # Ladda in policyn från pickle
    with open(pickle_file_path, 'rb') as f:
        policy = pickle.load(f)

    # Kontroll: Policyn ska ha .policy() metod
    if not hasattr(policy, 'policy'):
        raise ValueError("Loaded object does not have a .policy() method. Is this a TabularPolicy?")

    policy_dict = policy.policy()

    # Skriv till textfil
    with open(output_txt_path, 'w') as f:
        for info_state, action_probs in policy_dict.items():
            f.write(f"Information State: {info_state}\n")
            for action, prob in action_probs:
                f.write(f"  Action {action}: Probability {prob:.4f}\n")
            f.write("\n")  # Separera states

    print(f"Policy exported to {output_txt_path}")


# Exempel på körning:
if __name__ == "__main__":
    pickle_file = "/Users/davidklasa/Documents/GitHub/Kexet/main/PKL_models/Graf2/CFR_model_Graf2.pkl"
    output_txt = "/Users/davidklasa/Documents/GitHub/Kexet/main/test/cfr_policy_output.txt"

    export_tabular_policy(pickle_file, output_txt)