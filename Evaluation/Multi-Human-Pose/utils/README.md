## Evaluation on MPII Human Pose Dataset

This README provides instructions on how to prepare your predictions using MATLAB for evaluation on MPII HumanPose Dataset. Predictions should emailed to *[leonid at mpi-inf.mpg.de]* and *[eldar at mpi-inf.mpg.de]*.

### Preliminaries
Download [evaluation toolkit](http://human-pose.mpi-inf.mpg.de/results/mpii_human_pose/evalMPII.zip).

### Multi-Person Pose Estimation

#### Evaluation protocol
- Evaluation is performed on groups of multiple people. One image may contain multiple groups.
- Each group is localized by computing `x1, y1, x2, y2` group boundaries from locations of all people in the group and cropping around those boundaries.
    ```
    pos = zeros(length(rect),2);
    for ridx = 1:length(rect)
        pos(ridx,:) = [rect(ridx).objpos.x rect(ridx).objpos.y];
    end
    x1 = min(pos(:,1)); y1 = min(pos(:,2)); x2 = max(pos(:,1)); y2 = max(pos(:,2));
    ```
- Scale of each group is computed as an average scale of all people in the group.
    ```
    scale = zeros(length(rect),2);
    for ridx = 1:length(rect)
        scale(ridx) = rect(ridx).scale; 	
    end
    scaleGroup = mean(scale);
    ```
- Using ground truth number of people is **not allowed**.
- Using approximate location of each person while estimating person's pose is **not allowed**.
- Using individual scale of each person while estimating person's pose is **not allowed**.

#### Preparing predictions
1. Extract testing annotation list structure from the entire annotation list
    ```
    annolist_test = RELEASE.annolist(RELEASE.img_train == 0);
    ```
2. Extract groups of people using `getMultiPersonGroups.m` function from evaluation toolkit
    ```
    load('groups_v12.mat','groups');
    [imgidxs_multi_test,rectidxs_multi_test] = getMultiPersonGroups(groups,RELEASE,false);
    ```
    where `imgidxs_multi_test` are image IDs containing groups and `rectidxs_multi_test` are rectangle IDs of people in each group.
3. Split testing images into groups
    ```
    pred = annolist_test(imgidxs_multi_test);
    ```
4. Set predicted `x_pred, y_pred` coordinates and prediction `score` for each body joint
    ```
    pred(imgidx).annorect(ridx).annopoints.point(pidx).x = x_pred;
    pred(imgidx).annorect(ridx).annopoints.point(pidx).y = y_pred;
    pred(imgidx).annorect(ridx).annopoints.point(pidx).score = score;
    ```
5. Save predictions into `pred_keypoints_mpii_multi.mat`
    ```
    save('pred_keypoints_mpii_multi.mat','pred');
    ```

#### Evaluation Script
Evaluation is performed by using `evaluateAP.m` function

### Single Person Pose Estimation

#### Evaluation protocol
- Evaluation is performed on sufficiently separated people.
- Using approximate location and scale of each person **is allowed**
    ```
    pos = [rect(ridx).objpos.x rect(ridx).objpos.y];
    scale = rect(ridx).scale;
    ```

#### Preparing predictions
1. Extract testing annotation list structure from the entire annotation list:
    ```
    annolist_test = annolist(RELEASE.img_train == 0);
    ```
2. Extract image IDs and rectangle IDs of single persons
    ```
    rectidxs = RELEASE.single_person(RELEASE.img_train == 0);
    ```
3. Set predicted `x_pred, y_pred` coordinates for each body joint of single persons
    ```
    pred = annolist_test;
    pred(imgidx).annorect(ridx).annopoints.point(pidx).x = x_pred;
    pred(imgidx).annorect(ridx).annopoints.point(pidx).y = y_pred;
    ```
4. Save predictions into `pred_keypoints_mpii.mat`
    ```
    save('pred_keypoints_mpii.mat','pred');
    ``` 

#### Evaluation Script
Evaluation is performed by using `evaluatePCKh.m` function