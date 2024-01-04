"""
@author: Lobster
@software: PyCharm
@file: ensemble_and_output.py
@time: 2023/12/3 13:44
"""
import numpy as np
import pandas as pd


weight_list = [0.0780388463272377, 0.09593614331351352, 0.04988582106809258, 0.05180601878125834, 0.0336505835297281, 0.04977325999473468, 0.02167333680109353, 0.08157792273907567, 0.02367900558614327, 0.023602493712479185, 0.0174248056046096, 0.00024029657644931746, 0.10118405203265712,  0.03998276587482538, 0.07975472120643243, 0.02779590283650666, 0.03705624270273973, 0.008928753989775407]

df = pd.read_pickle(r'D:\RumorDetection\data\test_data\test_without_user.pkl')

id_without_user_list = df['id'].tolist()

model1_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens1\ens1.pkl')

model2_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens2\ens2.pkl')

model3_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens3\ens3.pkl')

model4_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens4\ens4.pkl')

model5_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens5\ens5.pkl')

model6_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens6\ens6.pkl')

model7_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens7\ens7.pkl')

model8_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens8\ens8.pkl')

model9_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens9\ens9.pkl')

model10_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens10\ens10.pkl')

model11_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens11\ens11.pkl')

model12_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens12\ens12.pkl')

model13_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens13\ens13.pkl')

model14_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens14\ens14.pkl')

model15_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens15\ens15.pkl')

model16_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens16\ens16.pkl')

model17_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens17\ens17.pkl')

model18_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens18\ens18.pkl')

model19_pred = pd.read_pickle(r'D:\RumorDetection\ensemble_and_output_results\ensembled_models\ens19\ens19.pkl')

output1 = model1_pred['output_1'].tolist()

output1 = [x[0] for x in output1]


output2 = model2_pred['output_2'].tolist()

output2 = [x[0] for x in output2]


output3 = model3_pred['output_3'].tolist()

output3 = [x[0] for x in output3]


output4 = model4_pred['output_4'].tolist()

output4 = [x[0] for x in output4]


output5 = model5_pred['output_5'].tolist()

output6 = model6_pred['output_6'].tolist()


output7 = model7_pred['output_7'].tolist()

output7 = [x[0] for x in output7]


output8 = model8_pred['output_8'].tolist()


output9 = model9_pred['output_9'].tolist()

output9 = [x[0] for x in output9]


output10 = model10_pred['output_10'].tolist()

output10 = [x[0] for x in output10]

output11 = model11_pred['output_11'].tolist()

output12 = model12_pred['output_12'].tolist()

output13 = model13_pred['output_13'].tolist()

output14 = model14_pred['output_18'].tolist()

output14 = [x[0] for x in output14]

output15 = model15_pred['output_19'].tolist()

output15 = [x[0] for x in output15]

output16 = model16_pred['output_20'].tolist()

output16 = [x[0] for x in output16]

output17 = model17_pred['output_new_1'].tolist()

output17 = [x[0] for x in output17]

output18 = model18_pred['output_new_2'].tolist()

output18 = [x[0] for x in output18]

output19 = model19_pred['pred'].tolist()

mask = model19_pred['mask'].tolist()

inverted_mask = model19_pred['inverted_mask'].tolist()

output_ens_all = []

output_ens_all.append(output1)
output_ens_all.append(output2)
output_ens_all.append(output3)
output_ens_all.append(output4)
output_ens_all.append(output5)
output_ens_all.append(output6)
output_ens_all.append(output7)
output_ens_all.append(output8)
output_ens_all.append(output9)
output_ens_all.append(output10)
output_ens_all.append(output11)
output_ens_all.append(output12)
output_ens_all.append(output13)
output_ens_all.append(output14)
output_ens_all.append(output15)
output_ens_all.append(output16)
output_ens_all.append(output17)
output_ens_all.append(output18)
output_ens_all.append(output19)


id_with_user_list = model1_pred['id'].tolist()

id_list = id_with_user_list + id_without_user_list

output = 0

for i in range(len(weight_list)):
    output += np.array(output_ens_all[i]) * np.array(weight_list[i])

output = output.tolist()

binary_output = [1 if x > 0.5 else 0 for x in output]

binary_output = binary_output + [1] * 24

binary_output = (np.array(binary_output) * np.array(mask) + np.array(output_ens_all[18]) * np.array(inverted_mask)).tolist()

output_df = pd.DataFrame()

output_df['id'] = id_list

output_df['label'] = binary_output

output_df.to_csv(r'D:\RumorDetection\submit\submission.csv',sep='\t',index=False)

print('结果生成完毕')


