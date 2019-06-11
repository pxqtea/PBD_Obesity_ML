
def ml_grid(data_train, data_valid, mode, method):
    ## Get the training data/label
    #mode = 2 
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=1)
    print("Generating inputs... ")
    data_all, label_all, ids_all = data_generator_reader(data_train, mode)  
    ### split the data into 80/20 train_test
    data, data_vd, label, label_vd = train_test_split(data_all, label_all, test_size=0.2, random_state=0)

    ## can use GridSearchCV to select the best set of parameters and make prediction on the CV sets
    #print(len(data), len(label))
    
    skf = StratifiedKFold(n_splits = cvsplits)
    scoring = ['precision_macro', 'recall_macro', 'accuracy','roc_auc']
    clf = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, \
          subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

    pipeline = Pipeline([  ('featuresel', SelectFromModel(clf)),
              ('classifer', clf) ], memory=memory)

    max_depth_options = range(4,7,1)   ## fine tuning in step of 1
    min_child_weight_options = range(2,4,1)  ## fine tuning in step of 1
    gamma_options = [i/10.0 for i in range(0,5)]
    subsample_options = [i/10.0 for i in range(6,10)]  ## fine tuning in step of 0.05
    colsample_bytree_options = [i/10.0 for i in range(6,10)]  ## fine tuning in step of 0.05
    reg_alpha_options = [1e-5, 1e-2, 0.1, 1, 100]
    learning_rate = 0.01
    n_estimators = 5000
    rfecv_step_options = [1, 0.01, 0.05, 0.1, 0.2]
    k_options = [10, 20, 50, 100]
    percentile_options = [10, 20, 30, 40, 50]

    ## tuning tree based parameters (max_depth and min_child_weight) with fixed learning rate and number of estimators
    parameters_1 = [
        {
            'featuresel': [RFECV(estimator=clf, scoring='roc_auc')],
            'featuresel__step': rfecv_step_options,     # split criterion
            'classifer__max_depth': max_depth_options,
            'classifer__min_child_weight': min_child_weight_options,
        },
    ]
   
    parameters_2 = [
        {
            'featuresel': [RFECV(estimator=clf, scoring='roc_auc')],
            'featuresel__step': rfecv_step_options,     # split criterion
            'classifer__gamma': gamma_options,
        },
        {
            'featuresel': [SelectPercentile(f_regression)],   ## chi2 is Good performance for sparse feature, but requires non-negative values
            'featuresel__percentile': percentile_options,
            'classifer__gamma': gamma_options,
        } ,
    ]
 
    parameters_3 = [
        {
            'featuresel': [RFECV(estimator=clf, scoring='roc_auc')],
            'featuresel__step': rfecv_step_options,     # split criterion
            'classifer__subsample': subsample_options,
            'classifer__colsample_bytree': colsample_bytree_options,
        },
        {
            'featuresel': [SelectPercentile(f_regression)],   ## chi2 is Good performance for sparse feature, but requires non-negative values
            'featuresel__percentile': percentile_options,
            'classifer__subsample': subsample_options,
            'classifer__colsample_bytree': colsample_bytree_options,
        } ,
    ]

    parameters_4 = [
        {
            'featuresel': [SelectFromModel(clf)],
            'classifer__reg_alpha': reg_alpha_options,
        },
        {
            'featuresel': [RFECV(estimator=clf, scoring='roc_auc')],
            'featuresel__step': rfecv_step_options,     # split criterion
            'classifer__reg_alpha': reg_alpha_options,
        },
        {
            'featuresel': [SelectPercentile(mutual_info_regression)],   ## chi2 is Good performance for sparse feature, but requires non-negative values
            'featuresel__percentile': percentile_options,
            'classifer__reg_alpha': reg_alpha_options,
        } ,
        {
            'featuresel': [SelectPercentile(f_regression)],   ## chi2 is Good performance for sparse feature, but requires non-negative values
            'featuresel__percentile': percentile_options,
            'classifer__reg_alpha': reg_alpha_options,
        } ,
    ]


    # find the best parameters for both the feature extraction and the classifier
    grid_search = GridSearchCV(pipeline, parameters_1, n_jobs=njobs, verbose=1, cv = skf, scoring = 'roc_auc' )
    
    #print("Performing grid search...")
    #print("pipeline:", [name for name, _ in pipeline.steps])
    #print("parameters:")
    t0 = time()
    print(len(data), len(label))
    grid_search.fit(data, label)
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Best score: %0.3f" % grid_search.best_score_)
    print()
    print("Best parameters set:")
    print(grid_search.best_estimator_.get_params())

    grid_search = GridSearchCV(grid_search.best_estimator_, parameters_2, n_jobs=njobs, verbose=1, cv = skf, scoring = 'roc_auc' )
    t0 = time()
    print(len(data), len(label))
    grid_search.fit(data, label)
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Best score: %0.3f" % grid_search.best_score_)
    print()
    print("Best parameters set:")
    print(grid_search.best_estimator_.get_params())

    grid_search = GridSearchCV(grid_search.best_estimator_, parameters_3, n_jobs=njobs, verbose=1, cv = skf, scoring = 'roc_auc' )
    t0 = time()
    print(len(data), len(label))
    grid_search.fit(data, label)
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Best score: %0.3f" % grid_search.best_score_)
    print()
    print("Best parameters set:")
    print(grid_search.best_estimator_.get_params())

    grid_search = GridSearchCV(grid_search.best_estimator_, parameters_4, n_jobs=njobs, verbose=1, cv = skf, scoring = 'roc_auc' )
    t0 = time()
    print(len(data), len(label))
    grid_search.fit(data, label)
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Best score: %0.3f" % grid_search.best_score_)
    print()
    print("Best parameters set:")
    print(grid_search.best_estimator_.get_params())

