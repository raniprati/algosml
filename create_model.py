from striprtf.striprtf import rtf_to_text
import json
import pathlib
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection
import sklearn.tree
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.neighbors
import sklearn.svm


def read_modelinfo_json(modelinfofile):
    with open(modelinfofile, "r") as infile:
        file_extension = pathlib.Path(modelinfofile).suffix
        if file_extension == ".rtf":
            rtfcontent = infile.read()
            textcontent = rtf_to_text(rtfcontent)
            modeldic = json.loads(textcontent.strip())
            #print(type(modeldic))
        elif file_extension == ".json":
            modeldic = json.load(infile)
            #print(type(modeldic))
    infile.close()
    with open("modelinfo.json", "w") as mfnm:
        json.dump(modeldic,mfnm)
    return modeldic

def show_session_info(infodic):
    sessiondic = infodic["design_state_data"]["session_info"]
    for item in sessiondic:
        print(item, " : ", sessiondic[item])

def get_target_type(infodic):
    targetval = infodic["design_state_data"]["target"]["target"]
    typeval = infodic["design_state_data"]["target"]["type"]
    return targetval, typeval

def get_metrics_info(infodic):
    metricsdic = infodic["design_state_data"]["metrics"]
    return metricsdic

def get_train_info(infodic):
    traindic = infodic["design_state_data"]["train"]
    return traindic

def feature_handling(indframe, modelinfodic):
    for featurename in indframe.columns:
        if modelinfodic["design_state_data"]["feature_handling"][featurename]["is_selected"]:
            #print(featurename)
            if "impute_value" in modelinfodic["design_state_data"]["feature_handling"][featurename]["feature_details"]:
                imputeval = modelinfodic["design_state_data"]["feature_handling"][featurename]["feature_details"]["impute_value"]
                #print(imputeval)
                indframe[featurename].fillna(imputeval, inplace=True)
            if "text_handling" in modelinfodic["design_state_data"]["feature_handling"][featurename]["feature_details"]:
                le = sklearn.preprocessing.LabelEncoder()
                indframe[featurename] = le.fit_transform(indframe[featurename])
                indframe[featurename] = indframe[featurename] + 1
        else:
            indframe.drop(featurename, axis=1, inplace=True)
    #print(indframe.head())
    return indframe

def feature_generation(indframe, modelinfodic):
    featuredic = modelinfodic["design_state_data"]["feature_generation"]
    featurelist = indframe.columns
    scalertransXin = pd.DataFrame(indframe)
    #print(scalertransXin.columns)
    linearXin = pd.DataFrame()
    polytran = pd.DataFrame()
    pairtran = pd.DataFrame()
    for val_feature_generation in featuredic:
        if val_feature_generation == "linear_scalar_type":
            for featurenm in featurelist:
                detaildic = modelinfodic["design_state_data"]["feature_handling"][featurenm]["feature_details"]
                if "rescaling" in detaildic:
                    if detaildic["rescaling"] != "No rescaling":
                        scaler = sklearn.preprocessing.RobustScaler().fit(indframe[featurenm])
                        modXinarray = scaler.transform(indframe[featurenm])
                        scaledframe = pd.DataFrame(modXinarray, columns=featurenm)
                        scalertransXin[featurenm] = scaledframe
                        #print(scalertransXin.columns)
        elif val_feature_generation == "linear_interactions":
            col1 = featuredic[val_feature_generation][0][0]
            col2 = featuredic[val_feature_generation][0][1]
            if (col1 in featurelist) and (col2 in featurelist):
                detaildic1 = modelinfodic["design_state_data"]["feature_handling"][col1]["feature_details"]
                detaildic2 = modelinfodic["design_state_data"]["feature_handling"][col2]["feature_details"]
                if ("make_derived_feats" in detaildic1) and ("make_derived_feats" in detaildic2):
                    permcol1 = detaildic1["make_derived_feats"]
                    permcol2 = detaildic2["make_derived_feats"]
                    if permcol1 and permcol2:
                        linearXin = linearXin.assign(lin_interaction=indframe[col1]*indframe[col2])
        elif val_feature_generation == "polynomial_interactions":
            pair1 = featuredic[val_feature_generation][0]
            pair2 = featuredic[val_feature_generation][1]
            valpair1 = pair1.split("/")
            #print(pair1, " : ", valpair1, " : ", valpair1[0], " : ", valpair1[1])
            if (valpair1[0] in featurelist) and (valpair1[1] in featurelist):
                detaildic1 = modelinfodic["design_state_data"]["feature_handling"][valpair1[0]]["feature_details"]
                detaildic2 = modelinfodic["design_state_data"]["feature_handling"][valpair1[1]]["feature_details"]
                if ("make_derived_feats" in detaildic1) and ("make_derived_feats" in detaildic2):
                    perm1 = detaildic1["make_derived_feats"]
                    perm2 = detaildic2["make_derived_feats"]
                    if perm1 and perm2:
                        polytran = polytran.assign(poly_interaction1=indframe[valpair1[0]]/indframe[valpair1[1]])
            valpair2 = pair2.split("/")
            # print(pair1, " : ", valpair1, " : ", valpair1[0], " : ", valpair1[1])
            if (valpair2[0] in featurelist) and (valpair2[1] in featurelist):
                detaildic1 = modelinfodic["design_state_data"]["feature_handling"][valpair2[0]]["feature_details"]
                detaildic2 = modelinfodic["design_state_data"]["feature_handling"][valpair2[1]]["feature_details"]
                if ("make_derived_feats" in detaildic1) and ("make_derived_feats" in detaildic2):
                    perm1 = detaildic1["make_derived_feats"]
                    perm2 = detaildic2["make_derived_feats"]
                    if perm1 and perm2:
                        polytran = polytran.assign(poly_interaction1=indframe[valpair2[0]] / indframe[valpair2[1]])
        elif val_feature_generation == "explicit_pairwise_interactions":
            pair1 = featuredic[val_feature_generation][0]
            pair2 = featuredic[val_feature_generation][1]
            valpair1 = pair1.split("/")
            # print(pair1, " : ", valpair1, " : ", valpair1[0], " : ", valpair1[1])
            if (valpair1[0] in featurelist) and (valpair1[1] in featurelist):
                detaildic1 = modelinfodic["design_state_data"]["feature_handling"][valpair1[0]]["feature_details"]
                detaildic2 = modelinfodic["design_state_data"]["feature_handling"][valpair1[1]]["feature_details"]
                if ("make_derived_feats" in detaildic1) and ("make_derived_feats" in detaildic2):
                    perm1 = detaildic1["make_derived_feats"]
                    perm2 = detaildic2["make_derived_feats"]
                    if perm1 and perm2:
                        pairtran = pairtran.assign(pair_interaction1=indframe[valpair1[0]]/indframe[valpair1[1]])
            valpair2 = pair2.split("/")
            # print(pair1, " : ", valpair1, " : ", valpair1[0], " : ", valpair1[1])
            if (valpair2[0] in featurelist) and (valpair2[1] in featurelist):
                detaildic1 = modelinfodic["design_state_data"]["feature_handling"][valpair2[0]]["feature_details"]
                detaildic2 = modelinfodic["design_state_data"]["feature_handling"][valpair2[1]]["feature_details"]
                if ("make_derived_feats" in detaildic1) and ("make_derived_feats" in detaildic2):
                    perm1 = detaildic1["make_derived_feats"]
                    perm2 = detaildic2["make_derived_feats"]
                    if perm1 and perm2:
                        pairtran = pairtran.assign(pair_interaction2=indframe[valpair2[0]]/indframe[valpair2[1]])
    transXin1 = pd.merge(scalertransXin, linearXin, left_index=True, right_index=True, how="outer")
    transXin2 = pd.merge(polytran, pairtran, left_index=True, right_index=True, how="outer")
    transXin = pd.merge(transXin1, transXin2, left_index=True, right_index=True, how="outer")
    #print(transXin.columns)
    #print(transXin.size)
    #print(transXin.shape)
    #print(transXin.head())
    return transXin


def feature_reduction(indframe, modelinfodic, labelcolname, proctype):
    Yin = pd.DataFrame(indframe[labelcolname])
    Xin = indframe.drop([labelcolname], axis=1)
    featuredmethod = modelinfodic["design_state_data"]["feature_reduction"]["feature_reduction_method"]
    featuredic = {}
    if featuredmethod in modelinfodic["design_state_data"]["feature_reduction"]:
        featuredic["feature_reduction_method"] = featuredmethod
        for fkey in modelinfodic["design_state_data"]["feature_reduction"][featuredmethod]:
            featuredic[fkey] = modelinfodic["design_state_data"]["feature_reduction"][featuredmethod][fkey]
    else:
        featuredic = modelinfodic["design_state_data"]["feature_reduction"]
    if featuredic["feature_reduction_method"] == "Tree-based":
        #print(featuredic)
        #print(Xin.columns)
        if proctype == "regression":
            model = sklearn.ensemble.RandomForestRegressor(n_estimators= int(featuredic["num_of_trees"]), max_depth=int(featuredic["depth_of_trees"]))
            model.fit(Xin, Yin[labelcolname])
        else:
            model = sklearn.ensemble.ExtraTreesClassifier(n_estimators=int(featuredic["num_of_trees"]), max_depth=int(featuredic["depth_of_trees"]))
            model.fit(Xin, Yin[labelcolname])
        feat_implist = list(model.feature_importances_)
        #print(feat_implist)
        sorted_feat_implist = feat_implist
        sorted_feat_implist.sort()
        i = 0
        valindx = len(sorted_feat_implist) - 1
        collist = []
        while (i <= int(featuredic["num_of_features_to_keep"])) and (valindx >=0):
            colindx = feat_implist.index(sorted_feat_implist[i])
            collist.append(colindx)
            valindx -= 1
            i += 1
        #print(collist)
        transformXin = Xin.iloc[:, collist]
        return transformXin, Yin
    elif featuredic["feature_reduction_method"] == "Correlation with target":
        return Yin, Yin
    elif featuredic["feature_reduction_method"] == "Principal Component Analysis":
        #print(featuredic)
        model = sklearn.decomposition.PCA(n_components=featuredic["num_of_features_to_keep"])
        transformXin = model.fit_transform(Xin)
        #print(transformXin.columns)
        return transformXin, Yin
    elif featuredic["feature_reduction_method"] == "No Reduction":
        return Xin, Yin

def get_hyperparameters(infodic):
    hyperdic = infodic["design_state_data"]["hyperparameters"]
    return hyperdic

def get_weighting_stratergy(infodic):
    weightingdic = infodic["design_state_data"]["weighting_stratergy"]
    return weightingdic

def get_probability_calibration(infodic):
    probabilitydic = infodic["design_state_data"]["probability_calibration"]
    return probabilitydic

def run_algo(modeldic, processtype, Xtrain, Xtest, Ytrain, Ytest):
    algodic = modeldic["design_state_data"]["algorithms"]
    hyperparamdic = get_hyperparameters(modeldic)
    for alognm in algodic:
        paramgrid = {}
        algoparamdic = modeldic["design_state_data"]["algorithms"][alognm]
        runcriteria1 = algoparamdic["model_name"].lower().find(processtype.lower())
        runcriteria2 = alognm.lower().find(processtype.lower())
        runcriteria3 = algoparamdic["model_name"].lower().find("Regressor".lower())
        runcriteria4 = alognm.lower().find("Regressor".lower())
        runcriteria = (runcriteria1 != -1) or (runcriteria2 != -1) or (runcriteria3 != -1) or (runcriteria4 != -1)
        if alognm == "RandomForestClassifier":
            model = sklearn.ensemble.RandomForestClassifier()
            paramgrid["n_estimators"] = [algoparamdic["min_trees"], algoparamdic["max_trees"]]
            paramgrid["max_depth"] = [algoparamdic["min_depth"], algoparamdic["max_depth"]]
            paramgrid["max_leaf_nodes"] = [algoparamdic["min_trees"], algoparamdic["max_trees"]]
            paramgrid["min_samples_leaf"] = [algoparamdic["min_samples_per_leaf_min_value"],
                                             algoparamdic["min_samples_per_leaf_max_value"]]
        if alognm == "RandomForestRegressor":
            model = sklearn.ensemble.RandomForestRegressor()
            paramgrid["n_estimators"] = [algoparamdic["min_trees"], algoparamdic["max_trees"]]
            paramgrid["max_depth"] = [algoparamdic["min_depth"], algoparamdic["max_depth"]]
            paramgrid["max_leaf_nodes"] = [algoparamdic["min_trees"], algoparamdic["max_trees"]]
            paramgrid["min_samples_leaf"] = [algoparamdic["min_samples_per_leaf_min_value"], algoparamdic["min_samples_per_leaf_max_value"]]
        if alognm == "GBTClassifier":
            model = sklearn.ensemble.GradientBoostingClassifier()
        if alognm == "GBTRegressor":
            model = sklearn.ensemble.GradientBoostingRegressor()
        if alognm == "DecisionTreeRegressor":
            model = sklearn.tree.DecisionTreeRegressor()
        if alognm == "DecisionTreeClassifier":
            model = sklearn.tree.DecisionTreeClassifier()
        if alognm == "LinearRegression":
            model = sklearn.linear_model.LinearRegression()
        if alognm == "LogisticRegression":
            model = sklearn.linear_model.LogisticRegression()
        if alognm == "RidgeRegression":
            model = sklearn.linear_model.Ridge()
        if alognm == "LassoRegression":
            model = sklearn.linear_model.Lasso()
        if alognm == "ElasticNetRegression":
            model = sklearn.linear_model.ElasticNet()
        if alognm == "xg_boost":
            model = sklearn.ensemble.GradientBoostingClassifier()
        if alognm == "SVM":
            model = sklearn.svm.SVR()
        if alognm == "SGD":
            model = sklearn.linear_model.SGDClassifier()
        if alognm == "KNN":
            model = sklearn.neighbors.KNeighborsClassifier()
        if alognm == "extra_random_trees":
            model = sklearn.ensemble.ExtraTreesClassifier()
        if alognm == "neural_network":
            model = sklearn.neural_network.MLPClassifier()
        if algoparamdic["is_selected"]:
            model.fit(Xtrain, Ytrain[Ytrain.columns[0]])
            model.predict(Xtest)
            modelscore = model.score(Xtest, Ytest)
            print("For model:",alognm, "accuracy score is: ", modelscore)
            #paradic = model.get_params()
            #print(paradic)
            if runcriteria and (len(paramgrid.keys())> 0):
                grid = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=paramgrid, cv=hyperparamdic["num_of_folds"])
                grid.fit(Xtrain, Ytrain[Ytrain.columns[0]])
                print(grid.cv_results_)
                print(grid.best_params_)
                print(grid.best_estimator_)
                print(grid.score(Xtrain, Ytrain))
                print(grid.score(Xtest, Ytest))

if __name__ == '__main__':
    modelinfofile = "algoparams_from_ui.json.rtf"
    datacsvfile = "iris.csv"
    inputdframe = pd.read_csv(datacsvfile)
    modeldic = read_modelinfo_json(modelinfofile)
    show_session_info(modeldic)
    traininfodic = get_train_info(modeldic)
    metricsinfodic = get_metrics_info(modeldic)
    hyperparamdic = get_hyperparameters(modeldic)
    weightinfodic = get_weighting_stratergy(modeldic)
    probinfodic = get_probability_calibration(modeldic)
    prosinputdframe = feature_handling(inputdframe, modeldic)
    transinputdframe = feature_generation(prosinputdframe, modeldic)
    targetcolname, algoprocesstype = get_target_type(modeldic)
    featredXinframe, Yinframe = feature_reduction(transinputdframe, modeldic, targetcolname, algoprocesstype)
    random_state_val = modeldic["design_state_data"]["train"]["random_seed"]
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(featredXinframe, Yinframe, random_state=random_state_val)
    run_algo(modeldic, algoprocesstype, X_train, X_test, Y_train, Y_test)
