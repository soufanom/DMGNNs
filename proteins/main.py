# main.py

from protfeaturegen import ProteinFeatureGenerator

def main():
    # Initialize the ProteinFeatureGenerator
    feature_generator = ProteinFeatureGenerator()

    # Example protein sequence
    protein_sequence = "MTARGLALGLLLLLLCPAQVFSQSCVWYGECGIAYGDKRYNCEYSGPPKPLPKDGYDLVQELCPGFFFGNVSLCCDVRQLQTLKDNLQLPLQFLSRCPSCFYNLLNLFCELTCSPRQSQFLNVTATEDYVDPVTNQTKTNVKELQYYVGQSFANAMYNACRDVEAPSSNDKALGLLCGKDADACNATNWIEYMFNKDNGQAPFTITPVFSDFPVHGMEPMNNATKGCDESVDEVTAPCSCQDCSIVCGPKPQPPPPPAPWTILGLDAMYVIMWITYMAFLLVFFGAFFAVWCYRKRYFVSEYTPIDSNIAFSVNASDKGEASCCDPVSAAFEGCLRRLFTRWGSFCVRNPGCVIFFSLVFITACSSGLVFVRVTTNPVDLWSAPSSQARLEKEYFDQHFGPFFRTEQLIIRAPLTDKHIYQPYPSGADVPFGPPLDIQILHQVLDLQIAIENITASYDNETVTLQDICLAPLSPYNTNCTILSVLNYFQNSHSVLDHKKGDDFFVYADYHTHFLYCVRAPASLNDTSLLHDPCLGTFGGPVFPWLVLGGYDDQNYNNATALVITFPVNNYYNDTEKLQRAQAWEKEFINFVKNYKNPNLTISFTAERSIEDELNRESDSDVFTVVISYAIMFLYISLALGHMKSCRRLLVDSKVSLGIAGILIVLSSVACSLGVFSYIGLPLTLIVIEVIPFLVLAVGVDNIFILVQAYQRDERLQGETLDQQLGRVLGEVAPSMFLSSFSETVAFFLGALSVMPAVHTFSLFAGLAVFIDFLLQITCFVSLLGLDIKRQEKNRLDIFCCVRGAEDGTSVQASESCLFRFFKNSYSPLLLKDWMRPIVIAIFVGVLSFSIAVLNKVDIGLDQSLSMPDDSYMVDYFKSISQYLHAGPPVYFVLEEGHDYTSSKGQNMVCGGMGCNNDSLVQQIFNAAQLDNYTRIGFAPSSWIDDYFDWVKPQSSCCRVDNITDQFCNASVVDPACVRCRPLTPEGKQRPQGGDFMRFLPMFLSDNPNPKCGKGGHAAYSSAVNILLGHGTRVGATYFMTYHTVLQTSADFIDALKKARLIASNVTETMGINGSAYRVFPYSVFYVFYEQYLTIIDDTIFNLGVSLGAIFLVTMVLLGCELWSAVIMCATIAMVLVNMFGVMWLWGISLNAVSLVNLVMSCGISVEFCSHITRAFTVSMKGSRVERAEEALAHMGSSVFSGITLTKFGGIVVLAFAKSQIFQIFYFRMYLAMVLLGATHGLIFLPVLLSYIGPSVNKAKSCATEERYKGTERERLLNF"

    # Generate one-hot encoding features
    one_hot_features = feature_generator.generate_onehot_features(protein_sequence)
    print("One-hot encoding features:")
    print(one_hot_features)

    # Generate amino acid composition (AAC) features
    aac_features = feature_generator.generate_aac_features(protein_sequence)
    print("\nAmino Acid Composition (AAC) features:")
    print(aac_features)

    # Generate physicochemical property features
    physico_features = feature_generator.generate_physicochemical_features(protein_sequence)
    print("\nPhysicochemical property features:")
    print(physico_features)

    # Generate combined features
    combined_features = feature_generator.generate_combined_features(protein_sequence)
    print("\nCombined features:")
    print(combined_features)

if __name__ == "__main__":
    main()