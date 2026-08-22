# Refinement-aware rotation audit

## Outcome

This train-fit, validation-only audit evaluated 55 matched hot-expert invocations over layers 0/4/20/39. It isolates gate/up refinement by holding down at exact Q4. Recovery is expert-output qenergy reconstruction, not task accuracy.

At 768 physical 512-byte pages, the best encoded method is `native_row_diagonal` with median isolated/output recovery 0.9065/0.9953. Native diagonal/exact-greedy reach 0.9065/0.9953 and 0.6823/0.9448, respectively.

The best layer-shared transform is `refinement_router_full_int2` at 0.8531/0.8460. The best physically encoded per-expert ceiling is `per_expert_separate_rank512_int8` at 0.9936/0.9931; its dense per-expert basis is charged below and is not deployable at one total bpw. The unencoded correction eigenbasis oracle reaches 0.9951/0.9943 at 768 action-count units and is not a physical bandwidth result.

For 95% exact gate/up correction recovery, per-expert joint INT4 needs a median 581 pages versus 875 for native diagonal, a 1.51x reduction. This is a basis ceiling, not a deployable win: the per-expert basis exceeds the one-bpw storage allowance. The strongest MSE-fitted four-level INT2 full-support endpoint is `refinement_uniform_full_int2` at 0.8785 median correction recovery; maximum 90%-target success across INT2 methods is 0.0%.

## Median validation frontier

| method                                         | physical_pages | isolated_recovery_median | expert_output_recovery_median | physically_encoded |
| ---------------------------------------------- | -------------- | ------------------------ | ----------------------------- | ------------------ |
| activation_pca_full_int2                       | 64             | 0.41301740571587864      | 0.4332214004659688            | True               |
| activation_pca_full_int2                       | 128            | 0.521081544805299        | 0.49153336119625957           | True               |
| activation_pca_full_int2                       | 256            | 0.6383469058976516       | 0.6249086486441484            | True               |
| activation_pca_full_int2                       | 384            | 0.7273273001023774       | 0.7159263230124673            | True               |
| activation_pca_full_int2                       | 512            | 0.7839923661510284       | 0.7527642358683417            | True               |
| activation_pca_full_int2                       | 688            | 0.8335977998791928       | 0.8195900429852262            | True               |
| activation_pca_full_int2                       | 728            | 0.840045955432853        | 0.8190006980339744            | True               |
| activation_pca_full_int2                       | 750            | 0.844846117488393        | 0.8336025596635689            | True               |
| activation_pca_full_int2                       | 752            | 0.8450359309154782       | 0.8307184096954645            | True               |
| activation_pca_full_int2                       | 758            | 0.8469129247020772       | 0.8292319405284095            | True               |
| activation_pca_full_int2                       | 760            | 0.8475966450591654       | 0.8255036448975742            | True               |
| activation_pca_full_int2                       | 764            | 0.8468342808479881       | 0.8307609520509098            | True               |
| activation_pca_full_int2                       | 768            | 0.8468884013303303       | 0.8257928569876807            | True               |
| activation_pca_full_int2                       | 1024           | 0.8689445847172733       | 0.8535787800227344            | True               |
| activation_pca_rank1024_int4                   | 64             | 0.4467386321277732       | 0.48381735672855786           | True               |
| activation_pca_rank1024_int4                   | 128            | 0.5082381048402611       | 0.5460910087073592            | True               |
| activation_pca_rank1024_int4                   | 256            | 0.5785945897151341       | 0.5932873093806783            | True               |
| activation_pca_rank1024_int4                   | 384            | 0.6221731792029306       | 0.634649434822031             | True               |
| activation_pca_rank1024_int4                   | 512            | 0.6435707534601103       | 0.6681969407574921            | True               |
| activation_pca_rank1024_int4                   | 688            | 0.6631304584122892       | 0.6883427255627533            | True               |
| activation_pca_rank1024_int4                   | 728            | 0.6651696105542          | 0.6810876302766151            | True               |
| activation_pca_rank1024_int4                   | 750            | 0.6653040640596544       | 0.6756398832184368            | True               |
| activation_pca_rank1024_int4                   | 752            | 0.6654326798049578       | 0.6754304814093256            | True               |
| activation_pca_rank1024_int4                   | 758            | 0.6660802327810403       | 0.677270182886557             | True               |
| activation_pca_rank1024_int4                   | 760            | 0.6661929142125718       | 0.6770335373103555            | True               |
| activation_pca_rank1024_int4                   | 764            | 0.6655958157311248       | 0.6682587151376223            | True               |
| activation_pca_rank1024_int4                   | 768            | 0.6658971069345706       | 0.6679924588709223            | True               |
| activation_pca_rank1024_int4                   | 1024           | 0.6681885362334368       | 0.6595073305693566            | True               |
| block_ajd_full_int2                            | 64             | 0.2870911812222223       | 0.2753774891301619            | True               |
| block_ajd_full_int2                            | 128            | 0.4037404329307718       | 0.4116261722160184            | True               |
| block_ajd_full_int2                            | 256            | 0.5604661055255398       | 0.5686136750160979            | True               |
| block_ajd_full_int2                            | 384            | 0.6705003819608331       | 0.6638509861313677            | True               |
| block_ajd_full_int2                            | 512            | 0.7441801022884869       | 0.7223008987395688            | True               |
| block_ajd_full_int2                            | 688            | 0.8111938373909799       | 0.7814160095420865            | True               |
| block_ajd_full_int2                            | 728            | 0.8201393360540461       | 0.7987542545828227            | True               |
| block_ajd_full_int2                            | 750            | 0.8241287706334112       | 0.8092609051935321            | True               |
| block_ajd_full_int2                            | 752            | 0.8240085256071279       | 0.8100220755027658            | True               |
| block_ajd_full_int2                            | 758            | 0.8255319274360955       | 0.8066170680319975            | True               |
| block_ajd_full_int2                            | 760            | 0.826035379481495        | 0.8061453066505313            | True               |
| block_ajd_full_int2                            | 764            | 0.827332552843353        | 0.8053245656909629            | True               |
| block_ajd_full_int2                            | 768            | 0.8284302856168433       | 0.8095764228791609            | True               |
| block_ajd_full_int2                            | 1024           | 0.8498203995779668       | 0.8239400596935516            | True               |
| block_ajd_rank1024_int4                        | 64             | 0.18736849702378677      | 0.11459661153635703           | True               |
| block_ajd_rank1024_int4                        | 128            | 0.2775016695043174       | 0.24309948091397848           | True               |
| block_ajd_rank1024_int4                        | 256            | 0.36923372752005057      | 0.38261950515089693           | True               |
| block_ajd_rank1024_int4                        | 384            | 0.4263549445101146       | 0.4402558205908318            | True               |
| block_ajd_rank1024_int4                        | 512            | 0.46098867253869413      | 0.48584134107905397           | True               |
| block_ajd_rank1024_int4                        | 688            | 0.4850183405598627       | 0.5355769574491596            | True               |
| block_ajd_rank1024_int4                        | 728            | 0.48566910928437923      | 0.5262160228418388            | True               |
| block_ajd_rank1024_int4                        | 750            | 0.4866344918073553       | 0.5407057461764306            | True               |
| block_ajd_rank1024_int4                        | 752            | 0.48705818518964994      | 0.541916796321354             | True               |
| block_ajd_rank1024_int4                        | 758            | 0.4877755432710087       | 0.5406979981681393            | True               |
| block_ajd_rank1024_int4                        | 760            | 0.48770158575506783      | 0.5355919997826313            | True               |
| block_ajd_rank1024_int4                        | 764            | 0.4879356934571568       | 0.5372455356846952            | True               |
| block_ajd_rank1024_int4                        | 768            | 0.4881451974119889       | 0.5353672999755124            | True               |
| block_ajd_rank1024_int4                        | 1024           | 0.49024782752894414      | 0.5338047444369958            | True               |
| identity_full_int2                             | 64             | 0.2745297597934587       | 0.3041658122101648            | True               |
| identity_full_int2                             | 128            | 0.3895244767079826       | 0.4332838910779855            | True               |
| identity_full_int2                             | 256            | 0.5449232705369276       | 0.5699037473416789            | True               |
| identity_full_int2                             | 384            | 0.647821092396123        | 0.621942901206707             | True               |
| identity_full_int2                             | 512            | 0.7028145215049446       | 0.7096079217131733            | True               |
| identity_full_int2                             | 688            | 0.7660620283495481       | 0.7386650380356403            | True               |
| identity_full_int2                             | 728            | 0.7739852554966616       | 0.7612037683064269            | True               |
| identity_full_int2                             | 750            | 0.777230158281661        | 0.7573979371000852            | True               |
| identity_full_int2                             | 752            | 0.7782610780142875       | 0.7618980501987868            | True               |
| identity_full_int2                             | 758            | 0.779236350555411        | 0.7628905762961743            | True               |
| identity_full_int2                             | 760            | 0.7790984318782382       | 0.7651019338942879            | True               |
| identity_full_int2                             | 764            | 0.7800020860890913       | 0.7706148336491694            | True               |
| identity_full_int2                             | 768            | 0.7806834755882782       | 0.7619899383240751            | True               |
| identity_full_int2                             | 1024           | 0.8075869919739702       | 0.7912551081587452            | True               |
| identity_paired_int4                           | 64             | 0.322781291721979        | 0.34132180510084653           | True               |
| identity_paired_int4                           | 128            | 0.4404571077978009       | 0.45814207625625114           | True               |
| identity_paired_int4                           | 256            | 0.5965101997496998       | 0.6077009902014596            | True               |
| identity_paired_int4                           | 384            | 0.6972681058899739       | 0.6985920301050788            | True               |
| identity_paired_int4                           | 512            | 0.7716527714494649       | 0.7686304358529368            | True               |
| identity_paired_int4                           | 688            | 0.8483181637917292       | 0.8479350446321033            | True               |
| identity_paired_int4                           | 728            | 0.8608433067109543       | 0.8603488382611895            | True               |
| identity_paired_int4                           | 750            | 0.8687208627472365       | 0.862043677463008             | True               |
| identity_paired_int4                           | 752            | 0.868705502919274        | 0.8575147530083433            | True               |
| identity_paired_int4                           | 758            | 0.8709480257897322       | 0.8565961134181079            | True               |
| identity_paired_int4                           | 760            | 0.8719009081420055       | 0.8579641288105084            | True               |
| identity_paired_int4                           | 764            | 0.8731022851199794       | 0.865845544509773             | True               |
| identity_paired_int4                           | 768            | 0.8735875099253212       | 0.8593845272100437            | True               |
| identity_paired_int4                           | 1024           | 0.9282842807135148       | 0.9300476471802938            | True               |
| native_row_diagonal                            | 64             | 0.19831062575750436      | 0.6870122306726649            | True               |
| native_row_diagonal                            | 128            | 0.32247083664853615      | 0.7689886293283317            | True               |
| native_row_diagonal                            | 256            | 0.5050156079829886       | 0.9008090396932517            | True               |
| native_row_diagonal                            | 384            | 0.6530569042337326       | 0.9535876027231659            | True               |
| native_row_diagonal                            | 512            | 0.758505500610261        | 0.9780039714561422            | True               |
| native_row_diagonal                            | 688            | 0.8623630473533078       | 0.9914778459290426            | True               |
| native_row_diagonal                            | 728            | 0.8857211557741584       | 0.9933697184491476            | True               |
| native_row_diagonal                            | 750            | 0.8966152948318025       | 0.9942562202767988            | True               |
| native_row_diagonal                            | 752            | 0.8971399003543342       | 0.9945920093987382            | True               |
| native_row_diagonal                            | 758            | 0.9016438358723717       | 0.9949378545714275            | True               |
| native_row_diagonal                            | 760            | 0.901688004508411        | 0.994968926902252             | True               |
| native_row_diagonal                            | 764            | 0.9040882951500953       | 0.994809310993875             | True               |
| native_row_diagonal                            | 768            | 0.90646683022044         | 0.9952627044571666            | True               |
| native_row_diagonal                            | 1024           | 1.0                      | 1.0                           | True               |
| native_row_exact_greedy                        | 64             | 0.1732005141764128       | 0.839443124987434             | True               |
| native_row_exact_greedy                        | 128            | 0.2691418833492272       | 0.890526212815143             | True               |
| native_row_exact_greedy                        | 256            | 0.40097418698305176      | 0.9295355191151471            | True               |
| native_row_exact_greedy                        | 384            | 0.4917922926760907       | 0.9391658079216245            | True               |
| native_row_exact_greedy                        | 512            | 0.5545465864571589       | 0.9436727087468401            | True               |
| native_row_exact_greedy                        | 688            | 0.6275104194042261       | 0.9438720909812076            | True               |
| native_row_exact_greedy                        | 728            | 0.6533231462103969       | 0.9442445890551008            | True               |
| native_row_exact_greedy                        | 750            | 0.665754596746565        | 0.9442848493349704            | True               |
| native_row_exact_greedy                        | 752            | 0.6673855435479009       | 0.9444159190523919            | True               |
| native_row_exact_greedy                        | 758            | 0.6720437715086442       | 0.9448120610983987            | True               |
| native_row_exact_greedy                        | 760            | 0.6740820369334781       | 0.9448449467252954            | True               |
| native_row_exact_greedy                        | 764            | 0.6815507473379702       | 0.9448407399038187            | True               |
| native_row_exact_greedy                        | 768            | 0.6822790537997385       | 0.9447810509316692            | True               |
| native_row_exact_greedy                        | 1024           | 1.0                      | 1.0                           | True               |
| per_expert_joint_rank1024_fp32_oracle          | 64             | 0.4451797965502198       | 0.413779814843368             | False              |
| per_expert_joint_rank1024_fp32_oracle          | 128            | 0.6213385895899453       | 0.6071703825219311            | False              |
| per_expert_joint_rank1024_fp32_oracle          | 256            | 0.8118653337271546       | 0.7958508825554695            | False              |
| per_expert_joint_rank1024_fp32_oracle          | 384            | 0.9052901521504896       | 0.9033475947187537            | False              |
| per_expert_joint_rank1024_fp32_oracle          | 512            | 0.9557845362378102       | 0.9579437005999948            | False              |
| per_expert_joint_rank1024_fp32_oracle          | 688            | 0.988898694296578        | 0.9880902633802565            | False              |
| per_expert_joint_rank1024_fp32_oracle          | 728            | 0.992404427340882        | 0.9911926154434829            | False              |
| per_expert_joint_rank1024_fp32_oracle          | 750            | 0.994032995762957        | 0.9931883015111052            | False              |
| per_expert_joint_rank1024_fp32_oracle          | 752            | 0.9941641618070189       | 0.9932331107804451            | False              |
| per_expert_joint_rank1024_fp32_oracle          | 758            | 0.9945413220462812       | 0.9941719412336585            | False              |
| per_expert_joint_rank1024_fp32_oracle          | 760            | 0.9946631324547168       | 0.9941899850991434            | False              |
| per_expert_joint_rank1024_fp32_oracle          | 764            | 0.9949029234054099       | 0.994361292949955             | False              |
| per_expert_joint_rank1024_fp32_oracle          | 768            | 0.9951355401939939       | 0.9943486750748427            | False              |
| per_expert_joint_rank1024_fp32_oracle          | 1024           | 0.9999999471389167       | 0.9999999391005973            | False              |
| per_expert_joint_rank1024_int2                 | 64             | 0.4456198830590866       | 0.38874852946351435           | True               |
| per_expert_joint_rank1024_int2                 | 128            | 0.6214329306401625       | 0.5976729278954748            | True               |
| per_expert_joint_rank1024_int2                 | 256            | 0.784350875907086        | 0.7913553789052588            | True               |
| per_expert_joint_rank1024_int2                 | 384            | 0.8484675477493699       | 0.8492341038863873            | True               |
| per_expert_joint_rank1024_int2                 | 512            | 0.863141531721203        | 0.845803937767167             | True               |
| per_expert_joint_rank1024_int2                 | 688            | 0.863141531721203        | 0.845803937767167             | True               |
| per_expert_joint_rank1024_int2                 | 728            | 0.863141531721203        | 0.845803937767167             | True               |
| per_expert_joint_rank1024_int2                 | 750            | 0.863141531721203        | 0.845803937767167             | True               |
| per_expert_joint_rank1024_int2                 | 752            | 0.863141531721203        | 0.845803937767167             | True               |
| per_expert_joint_rank1024_int2                 | 758            | 0.863141531721203        | 0.845803937767167             | True               |
| per_expert_joint_rank1024_int2                 | 760            | 0.863141531721203        | 0.845803937767167             | True               |
| per_expert_joint_rank1024_int2                 | 764            | 0.863141531721203        | 0.845803937767167             | True               |
| per_expert_joint_rank1024_int2                 | 768            | 0.863141531721203        | 0.845803937767167             | True               |
| per_expert_joint_rank1024_int2                 | 1024           | 0.863141531721203        | 0.845803937767167             | True               |
| per_expert_joint_rank1024_int4                 | 64             | 0.43477612228418827      | 0.4295820555101526            | True               |
| per_expert_joint_rank1024_int4                 | 128            | 0.6084951133204927       | 0.5981617589330632            | True               |
| per_expert_joint_rank1024_int4                 | 256            | 0.7917432408457127       | 0.763867010106774             | True               |
| per_expert_joint_rank1024_int4                 | 384            | 0.8833633483543281       | 0.8717780344221069            | True               |
| per_expert_joint_rank1024_int4                 | 512            | 0.9328589488444946       | 0.9348225340281677            | True               |
| per_expert_joint_rank1024_int4                 | 688            | 0.9660444022119815       | 0.9595078111810099            | True               |
| per_expert_joint_rank1024_int4                 | 728            | 0.9693852402216017       | 0.9640655236123854            | True               |
| per_expert_joint_rank1024_int4                 | 750            | 0.971660137199904        | 0.9647526267885227            | True               |
| per_expert_joint_rank1024_int4                 | 752            | 0.9716494397706641       | 0.9645271175635421            | True               |
| per_expert_joint_rank1024_int4                 | 758            | 0.9721825558470725       | 0.9640622335987908            | True               |
| per_expert_joint_rank1024_int4                 | 760            | 0.9722731290187887       | 0.9647231200823375            | True               |
| per_expert_joint_rank1024_int4                 | 764            | 0.9723165629814848       | 0.9652790882183978            | True               |
| per_expert_joint_rank1024_int4                 | 768            | 0.9723170585202404       | 0.9648627169454407            | True               |
| per_expert_joint_rank1024_int4                 | 1024           | 0.9763674813553136       | 0.9732323244061909            | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 64             | 0.3714868681674349       | 0.46257156263723165           | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 128            | 0.5540020284733659       | 0.6102129433532364            | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 256            | 0.7526128004971333       | 0.8393870587906938            | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 384            | 0.8565358397011789       | 0.9125548249641602            | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 512            | 0.9154569974344744       | 0.9386992377049055            | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 688            | 0.9556171771481633       | 0.9675960192598662            | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 728            | 0.9599911782695888       | 0.9695050357087649            | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 750            | 0.9624653003127712       | 0.9706262913864956            | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 752            | 0.9628754408765423       | 0.9706270357710436            | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 758            | 0.9633407449375098       | 0.9705290663047762            | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 760            | 0.9635218688838274       | 0.970626487656753             | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 764            | 0.9639360770357978       | 0.9706258924245836            | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 768            | 0.964217514545921        | 0.9706925435336263            | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 1024           | 0.9696967992526981       | 0.9729559762300037            | True               |
| per_expert_separate_rank512_int8               | 64             | 0.39591045006787695      | 0.3859256980780349            | True               |
| per_expert_separate_rank512_int8               | 128            | 0.5705190884385704       | 0.5547600805073888            | True               |
| per_expert_separate_rank512_int8               | 256            | 0.7734400547991647       | 0.7886693384172435            | True               |
| per_expert_separate_rank512_int8               | 384            | 0.8828520288381284       | 0.8859313627427309            | True               |
| per_expert_separate_rank512_int8               | 512            | 0.9438301339773902       | 0.9422972909239558            | True               |
| per_expert_separate_rank512_int8               | 688            | 0.9851469122310302       | 0.9834886065234403            | True               |
| per_expert_separate_rank512_int8               | 728            | 0.9899065091570911       | 0.9881376178166565            | True               |
| per_expert_separate_rank512_int8               | 750            | 0.9920633268325009       | 0.9910702588614207            | True               |
| per_expert_separate_rank512_int8               | 752            | 0.9922368220448629       | 0.9905909341289423            | True               |
| per_expert_separate_rank512_int8               | 758            | 0.992744631477523        | 0.9917120291517768            | True               |
| per_expert_separate_rank512_int8               | 760            | 0.9929142780758852       | 0.9918373070411135            | True               |
| per_expert_separate_rank512_int8               | 764            | 0.9932299721726918       | 0.9923499139990439            | True               |
| per_expert_separate_rank512_int8               | 768            | 0.9935517852491189       | 0.9931330625446543            | True               |
| per_expert_separate_rank512_int8               | 1024           | 0.9999154289161569       | 0.9999169622551649            | True               |
| refinement_router_full_int2                    | 64             | 0.3140361313085468       | 0.34144961243852867           | True               |
| refinement_router_full_int2                    | 128            | 0.4400515389275529       | 0.42952277950113216           | True               |
| refinement_router_full_int2                    | 256            | 0.6039604710062988       | 0.579738448197332             | True               |
| refinement_router_full_int2                    | 384            | 0.7104349301028738       | 0.6945588431579595            | True               |
| refinement_router_full_int2                    | 512            | 0.778501771353673        | 0.7601518161099834            | True               |
| refinement_router_full_int2                    | 688            | 0.8348549795730124       | 0.8273742009310714            | True               |
| refinement_router_full_int2                    | 728            | 0.8451285384585038       | 0.833022534202627             | True               |
| refinement_router_full_int2                    | 750            | 0.8504777712779328       | 0.8364322794346466            | True               |
| refinement_router_full_int2                    | 752            | 0.8505266528225157       | 0.843743796134639             | True               |
| refinement_router_full_int2                    | 758            | 0.8510546959702237       | 0.8461586051522547            | True               |
| refinement_router_full_int2                    | 760            | 0.8515697603421508       | 0.8446120488766067            | True               |
| refinement_router_full_int2                    | 764            | 0.8525117651018994       | 0.8475286811079195            | True               |
| refinement_router_full_int2                    | 768            | 0.853056972534901        | 0.8459977576489559            | True               |
| refinement_router_full_int2                    | 1024           | 0.8758181264316749       | 0.872379040084299             | True               |
| refinement_router_rank1024_int4                | 64             | 0.29449423542340325      | 0.2958046522875263            | True               |
| refinement_router_rank1024_int4                | 128            | 0.40622616145429413      | 0.3999575514668672            | True               |
| refinement_router_rank1024_int4                | 256            | 0.5486330866269351       | 0.5608680810744997            | True               |
| refinement_router_rank1024_int4                | 384            | 0.6279056337299505       | 0.6314938423034264            | True               |
| refinement_router_rank1024_int4                | 512            | 0.6659137455651375       | 0.6759787588385744            | True               |
| refinement_router_rank1024_int4                | 688            | 0.6938864148668957       | 0.7015214496087008            | True               |
| refinement_router_rank1024_int4                | 728            | 0.6970472193075252       | 0.707827566114219             | True               |
| refinement_router_rank1024_int4                | 750            | 0.6984404125320857       | 0.7136703359342901            | True               |
| refinement_router_rank1024_int4                | 752            | 0.6982671382948658       | 0.7149835273024526            | True               |
| refinement_router_rank1024_int4                | 758            | 0.6985621342485139       | 0.7181586032942759            | True               |
| refinement_router_rank1024_int4                | 760            | 0.6988838819653662       | 0.7156585185655719            | True               |
| refinement_router_rank1024_int4                | 764            | 0.6991015808016541       | 0.7149814795072043            | True               |
| refinement_router_rank1024_int4                | 768            | 0.6991678484715955       | 0.7167620810049007            | True               |
| refinement_router_rank1024_int4                | 1024           | 0.7051370698293189       | 0.7115583128255907            | True               |
| refinement_uniform_full_int2                   | 64             | 0.2733213004185261       | 0.2756764027191164            | True               |
| refinement_uniform_full_int2                   | 128            | 0.3952997373509012       | 0.38939989397848673           | True               |
| refinement_uniform_full_int2                   | 256            | 0.5647146722251148       | 0.6044136202794109            | True               |
| refinement_uniform_full_int2                   | 384            | 0.6810265626445968       | 0.6907068912350962            | True               |
| refinement_uniform_full_int2                   | 512            | 0.7613177830313936       | 0.7706835897805124            | True               |
| refinement_uniform_full_int2                   | 688            | 0.8335752125000795       | 0.8469365682235701            | True               |
| refinement_uniform_full_int2                   | 728            | 0.8447309666845381       | 0.8580869956914533            | True               |
| refinement_uniform_full_int2                   | 750            | 0.8509792540220348       | 0.8370132488736943            | True               |
| refinement_uniform_full_int2                   | 752            | 0.8520965119003998       | 0.8462272437293236            | True               |
| refinement_uniform_full_int2                   | 758            | 0.8532420674909761       | 0.8499598086515247            | True               |
| refinement_uniform_full_int2                   | 760            | 0.85376351716475         | 0.8459297915194842            | True               |
| refinement_uniform_full_int2                   | 764            | 0.8538857446922805       | 0.8403116438201084            | True               |
| refinement_uniform_full_int2                   | 768            | 0.8544191116665726       | 0.8405692787466597            | True               |
| refinement_uniform_full_int2                   | 1024           | 0.8785353403079705       | 0.8702492066910552            | True               |
| refinement_uniform_rank1024_int4               | 64             | 0.2191392115728905       | 0.21821157423056936           | True               |
| refinement_uniform_rank1024_int4               | 128            | 0.31224480702966173      | 0.3065392389002969            | True               |
| refinement_uniform_rank1024_int4               | 256            | 0.41686174468590376      | 0.39147016773455967           | True               |
| refinement_uniform_rank1024_int4               | 384            | 0.48245112051768924      | 0.4500299014466179            | True               |
| refinement_uniform_rank1024_int4               | 512            | 0.525165108634787        | 0.5002108852220365            | True               |
| refinement_uniform_rank1024_int4               | 688            | 0.5484447184366688       | 0.5533696607145369            | True               |
| refinement_uniform_rank1024_int4               | 728            | 0.5495152905967537       | 0.5670927449816476            | True               |
| refinement_uniform_rank1024_int4               | 750            | 0.5530572155344466       | 0.5642145876890609            | True               |
| refinement_uniform_rank1024_int4               | 752            | 0.5524952007892427       | 0.561101685630214             | True               |
| refinement_uniform_rank1024_int4               | 758            | 0.5529898272276439       | 0.5543411946127288            | True               |
| refinement_uniform_rank1024_int4               | 760            | 0.5532083112167931       | 0.5545288528619796            | True               |
| refinement_uniform_rank1024_int4               | 764            | 0.5538364555240309       | 0.5544854022378793            | True               |
| refinement_uniform_rank1024_int4               | 768            | 0.5535409910711853       | 0.5598840505264164            | True               |
| refinement_uniform_rank1024_int4               | 1024           | 0.555562560991175        | 0.5540122214716996            | True               |

## Physical pages to 95% exact gate/up correction recovery

| method                                         | success_fraction | actions_median | actions_p90 | action_bpw_median  | physically_encoded |
| ---------------------------------------------- | ---------------- | -------------- | ----------- | ------------------ | ------------------ |
| per_expert_joint_rank1024_fp32_oracle          | 1.0              | 493.0          | 507.6       | nan                | False              |
| per_expert_separate_rank512_int8               | 1.0              | 531.0          | 545.0       | 0.69140625         | True               |
| per_expert_joint_rank1024_int4                 | 1.0              | 581.0          | 612.8       | 0.7565104166666666 | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 1.0              | 655.0          | 686.0       | 0.8528645833333334 | True               |
| native_row_diagonal                            | 1.0              | 875.0          | 922.6       | 1.1393229166666667 | True               |
| native_row_exact_greedy                        | 1.0              | 1001.0         | 1014.0      | 1.3033854166666667 | True               |
| identity_paired_int4                           | 1.0              | 1178.0         | 1257.4      | 1.5338541666666667 | True               |
| activation_pca_full_int2                       | 0.0              | nan            | nan         | nan                | True               |
| activation_pca_rank1024_int4                   | 0.0              | nan            | nan         | nan                | True               |
| block_ajd_full_int2                            | 0.0              | nan            | nan         | nan                | True               |
| block_ajd_rank1024_int4                        | 0.0              | nan            | nan         | nan                | True               |
| identity_full_int2                             | 0.0              | nan            | nan         | nan                | True               |
| per_expert_joint_rank1024_int2                 | 0.0              | nan            | nan         | nan                | True               |
| refinement_router_full_int2                    | 0.0              | nan            | nan         | nan                | True               |
| refinement_router_rank1024_int4                | 0.0              | nan            | nan         | nan                | True               |
| refinement_uniform_full_int2                   | 0.0              | nan            | nan         | nan                | True               |
| refinement_uniform_rank1024_int4               | 0.0              | nan            | nan         | nan                | True               |

## Diagonal versus exact fixed-coefficient greedy

| method                                | target_recovery | n | finite_fraction | diagonal_to_exact_ratio_median | diagonal_to_exact_ratio_p90 |
| ------------------------------------- | --------------- | - | --------------- | ------------------------------ | --------------------------- |
| block_ajd_full_int2                   | 0.9             | 4 | 0.0             | nan                            | nan                         |
| block_ajd_full_int2                   | 0.95            | 4 | 0.0             | nan                            | nan                         |
| block_ajd_full_int2                   | 0.99            | 4 | 0.0             | nan                            | nan                         |
| block_ajd_rank1024_int4               | 0.9             | 4 | 0.0             | nan                            | nan                         |
| block_ajd_rank1024_int4               | 0.95            | 4 | 0.0             | nan                            | nan                         |
| block_ajd_rank1024_int4               | 0.99            | 4 | 0.0             | nan                            | nan                         |
| identity_full_int2                    | 0.9             | 4 | 0.0             | nan                            | nan                         |
| identity_full_int2                    | 0.95            | 4 | 0.0             | nan                            | nan                         |
| identity_full_int2                    | 0.99            | 4 | 0.0             | nan                            | nan                         |
| identity_paired_int4                  | 0.9             | 4 | 1.0             | 1.7780663234403393             | 2.419547817047817           |
| identity_paired_int4                  | 0.95            | 4 | 1.0             | 1.4676958353954777             | 2.3125821650527536          |
| identity_paired_int4                  | 0.99            | 4 | 0.0             | nan                            | nan                         |
| per_expert_joint_rank1024_fp32_oracle | 0.9             | 4 | 1.0             | 1.0                            | 1.0                         |
| per_expert_joint_rank1024_fp32_oracle | 0.95            | 4 | 1.0             | 1.0                            | 1.0                         |
| per_expert_joint_rank1024_fp32_oracle | 0.99            | 4 | 1.0             | 1.0                            | 1.0                         |
| per_expert_joint_rank1024_int2        | 0.9             | 4 | 0.0             | nan                            | nan                         |
| per_expert_joint_rank1024_int2        | 0.95            | 4 | 0.0             | nan                            | nan                         |
| per_expert_joint_rank1024_int2        | 0.99            | 4 | 0.0             | nan                            | nan                         |
| per_expert_joint_rank1024_int4        | 0.9             | 4 | 1.0             | 1.0590528370804206             | 1.069752560269703           |
| per_expert_joint_rank1024_int4        | 0.95            | 4 | 1.0             | 1.0672546366576463             | 1.0723769808173478          |
| per_expert_joint_rank1024_int4        | 0.99            | 4 | 0.0             | nan                            | nan                         |

## Storage and transform cost

| method                                         | physically_encoded | budget_semantics        | action_payload_bytes | coordinates_per_action | representation_code_bytes | representation_scale_bytes | representation_bytes | basis_scope  | basis_storage_bytes | transform_macs | amortized_basis_bytes_per_expert | resident_metadata_bytes_per_expert | resident_metadata_equivalent_bpw | strict_one_bpw_action_pages |
| ---------------------------------------------- | ------------------ | ----------------------- | -------------------- | ---------------------- | ------------------------- | -------------------------- | -------------------- | ------------ | ------------------- | -------------- | -------------------------------- | ---------------------------------- | -------------------------------- | --------------------------- |
| activation_pca_full_int2                       | True               | physical_512_byte_pages | 512                  | 2                      | 524288                    | 8192                       | 532480               | layer_shared | 8388608             | 4194304        | 32768                            | 40960                              | 0.10416666666666667              | 688                         |
| activation_pca_rank1024_int4                   | True               | physical_512_byte_pages | 512                  | 1                      | 524288                    | 4096                       | 528384               | layer_shared | 4194304             | 2097152        | 16384                            | 20480                              | 0.052083333333333336             | 728                         |
| block_ajd_full_int2                            | True               | physical_512_byte_pages | 512                  | 2                      | 524288                    | 8192                       | 532480               | layer_shared | 135168              | 65536          | 528                              | 8720                               | 0.022176106770833332             | 750                         |
| block_ajd_rank1024_int4                        | True               | physical_512_byte_pages | 512                  | 1                      | 524288                    | 4096                       | 528384               | layer_shared | 133120              | 32768          | 520                              | 4616                               | 0.011739095052083334             | 758                         |
| identity_full_int2                             | True               | physical_512_byte_pages | 512                  | 2                      | 524288                    | 8192                       | 532480               | fixed        | 0                   | 0              | 0                                | 8192                               | 0.020833333333333332             | 752                         |
| identity_paired_int4                           | True               | physical_512_byte_pages | 512                  | 1                      | 1048576                   | 8192                       | 1056768              | fixed        | 0                   | 0              | 0                                | 8192                               | 0.020833333333333332             | 752                         |
| native_row_diagonal                            | True               | physical_512_byte_pages | 512                  | 1                      | 524288                    | 0                          | 524288               | native       | 0                   | 0              | 0                                | 0                                  | 0.0                              | 768                         |
| native_row_exact_greedy                        | True               | physical_512_byte_pages | 512                  | 1                      | 524288                    | 0                          | 524288               | native       | 0                   | 0              | 0                                | 0                                  | 0.0                              | 768                         |
| per_expert_joint_rank1024_fp32_oracle          | False              | action_count_only       | 4096                 | 1                      | 4194304                   | 0                          | 4194304              | per_expert   | 4194304             | 2097152        | 4194304                          | 4194304                            | 10.666666666666666               | 0                           |
| per_expert_joint_rank1024_int2                 | True               | physical_512_byte_pages | 512                  | 2                      | 262144                    | 4096                       | 266240               | per_expert   | 4194304             | 2097152        | 4194304                          | 4198400                            | 10.677083333333334               | 0                           |
| per_expert_joint_rank1024_int4                 | True               | physical_512_byte_pages | 512                  | 1                      | 524288                    | 4096                       | 528384               | per_expert   | 4194304             | 2097152        | 4194304                          | 4198400                            | 10.677083333333334               | 0                           |
| per_expert_output_jacobian_train_rank1024_int4 | True               | physical_512_byte_pages | 512                  | 1                      | 524288                    | 4096                       | 528384               | per_expert   | 4194304             | 2097152        | 4194304                          | 4198400                            | 10.677083333333334               | 0                           |
| per_expert_separate_rank512_int8               | True               | physical_512_byte_pages | 512                  | 1                      | 524288                    | 2048                       | 526336               | per_expert   | 4194304             | 2097152        | 4194304                          | 4196352                            | 10.671875                        | 0                           |
| refinement_router_full_int2                    | True               | physical_512_byte_pages | 512                  | 2                      | 524288                    | 8192                       | 532480               | layer_shared | 8388608             | 4194304        | 32768                            | 40960                              | 0.10416666666666667              | 688                         |
| refinement_router_rank1024_int4                | True               | physical_512_byte_pages | 512                  | 1                      | 524288                    | 4096                       | 528384               | layer_shared | 4194304             | 2097152        | 16384                            | 20480                              | 0.052083333333333336             | 728                         |
| refinement_uniform_full_int2                   | True               | physical_512_byte_pages | 512                  | 2                      | 524288                    | 8192                       | 532480               | layer_shared | 8388608             | 4194304        | 32768                            | 40960                              | 0.10416666666666667              | 688                         |
| refinement_uniform_rank1024_int4               | True               | physical_512_byte_pages | 512                  | 1                      | 524288                    | 4096                       | 528384               | layer_shared | 4194304             | 2097152        | 16384                            | 20480                              | 0.052083333333333336             | 728                         |

## Metadata-aware one-bpw control

This conservative control subtracts per-column scales and the per-expert share of a layer-shared basis from 393,216 bytes before assigning 512-byte actions. Per-expert dense bases exceed the entire one-bpw allowance and therefore receive zero strict actions.

| method                                         | strict_action_pages | evaluated_pages | output_recovery_median | physically_encoded |
| ---------------------------------------------- | ------------------- | --------------- | ---------------------- | ------------------ |
| activation_pca_full_int2                       | 688                 | 688             | 0.8195900429852262     | True               |
| activation_pca_rank1024_int4                   | 728                 | 728             | 0.6810876302766151     | True               |
| block_ajd_full_int2                            | 750                 | 750             | 0.8092609051935321     | True               |
| block_ajd_rank1024_int4                        | 758                 | 758             | 0.5406979981681393     | True               |
| identity_full_int2                             | 752                 | 752             | 0.7618980501987868     | True               |
| identity_paired_int4                           | 752                 | 752             | 0.8575147530083433     | True               |
| native_row_diagonal                            | 768                 | 768             | 0.9952627044571666     | True               |
| native_row_exact_greedy                        | 768                 | 768             | 0.9447810509316692     | True               |
| per_expert_joint_rank1024_fp32_oracle          | 0                   | 0               | 0.0                    | False              |
| per_expert_joint_rank1024_int2                 | 0                   | 0               | 0.0                    | True               |
| per_expert_joint_rank1024_int4                 | 0                   | 0               | 0.0                    | True               |
| per_expert_output_jacobian_train_rank1024_int4 | 0                   | 0               | 0.0                    | True               |
| per_expert_separate_rank512_int8               | 0                   | 0               | 0.0                    | True               |
| refinement_router_full_int2                    | 688                 | 688             | 0.8273742009310714     | True               |
| refinement_router_rank1024_int4                | 728                 | 728             | 0.707827566114219      | True               |
| refinement_uniform_full_int2                   | 688                 | 688             | 0.8469365682235701     | True               |
| refinement_uniform_rank1024_int4               | 728                 | 728             | 0.5670927449816476     | True               |

## Scope

- Shared and per-expert bases are fit without validation values; selection and summaries use validation only. Test rows are not admitted.
- Paired INT4 stores one transformed gate/up coordinate per 512-byte page. Full-rank INT2 uses an MSE-fitted symmetric four-level codebook and stores two fixed coordinate pairs per page; selectors rank these indivisible pages, not arbitrary half-pages. Separate INT8 stores one projection column per page. FP16 scales, basis storage, and transform MACs are charged separately.
- Dense per-expert transforms are ceilings, not deployable candidates. The block-AJD transform is the low-cost shared control.
- No H4 predictor, prefetch timing, causal replay, routing/logit agreement, token quality, perplexity, or downstream benchmark is evaluated.

Run scope: `train-fit validation-only matched hot-expert basis ceiling; exact gate/up Q2-to-Q4 residuals with Q4 down held fixed; no test, H4 predictor, causal replay, latency, logits, tokens, perplexity, or deployment claim`.
