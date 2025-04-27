package com.sjtu.KGStudy.abnormalDataRec.utils;

import com.sjtu.KGStudy.utils.PrefixesDel;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.update.UpdateAction;
import org.apache.jena.update.UpdateFactory;
import org.apache.jena.update.UpdateRequest;

public class DelHistoricalAbnormalLabel {
    public static void labelHistoricalObservationStatus(Model delModel, String observationType, boolean currentObsStatus){
        int updatedAbnormalIndex = -1;
        int hisAbnormalIndex = -1;
        try {
            hisAbnormalIndex = getHistoricalObservationStatus(delModel, observationType);
        } catch (CustomHisAbnormalIndexException e) {
            e.printStackTrace();
        }

        boolean updatedAbnormalIndexOption = true;
        if (currentObsStatus) {  // 如果当前观测为True（表示当前观测正常）
            if (hisAbnormalIndex != -1){  // 当前观测正常，历史观测存在异常记录，异常记录的abnormalIndex+1
                updatedAbnormalIndex = hisAbnormalIndex+1;
                if (updatedAbnormalIndex > 50) {  // 表示超出了用于重构历史数据的窗口范畴
                    updatedAbnormalIndex = -1;
                }
            }else {  // 当前观测正常，历史观测不存在异常记录，保持abnormalIndex为-1
                updatedAbnormalIndexOption = false;
            }
        } else {// 如果当前观测为False（表示当前观测异常），直接将置abnormalIndex为0
            updatedAbnormalIndex = 0;
        }

        if (updatedAbnormalIndexOption) {
            UpdateRequest request = UpdateFactory.create();
            String[] updateRequests = new String[2];
            StringBuffer stringBuffer = PrefixesDel.getPrefixes(delModel);
            String prefixesSen = stringBuffer.toString();

            String updateReqDel = "DELETE {" + observationType + " csm:hasAbnormalObsRecentIndex ?o.} \n "
                    + "WHERE {" + observationType + " csm:hasAbnormalObsRecentIndex ?o.}";

            String updateReqInsert = "INSERT {" + observationType + " csm:hasAbnormalObsRecentIndex \'" + updatedAbnormalIndex + "\'^^xsd:integer.} \n "
                    + "WHERE {}";

            // 将删除语句加入更新请求
            updateRequests[0] = prefixesSen + updateReqDel;

            updateRequests[1] = prefixesSen + updateReqInsert;

            for (String updateStr : updateRequests) {
                request.add(updateStr);
            }

            UpdateAction.execute(request, delModel);
        }
    }


    public static int getHistoricalObservationStatus(Model delModel, String observationType) throws CustomHisAbnormalIndexException {
        int abnormalIndex = 0;

        StringBuffer stringBufferQuery = PrefixesDel.getPrefixes(delModel);
        String queryBody = "SELECT ?abnormalIndex \n" +
                "WHERE{" + observationType + " csm:hasAbnormalObsRecentIndex ?abnormalIndex.}";

        stringBufferQuery.append(queryBody);
        Query querySPARQL = QueryFactory.create(stringBufferQuery.toString());

        // 执行查询并获取结果
        try (QueryExecution qexec = QueryExecutionFactory.create(querySPARQL, delModel)) {
            ResultSet rs = qexec.execSelect();

            if (rs.hasNext()){  // 这里只有一个查询结果
                String indexStr = rs.nextSolution().get("abnormalIndex").toString();
                // 查找 "^^" 的位置
                int index = indexStr.indexOf("^^");

                // 截取 "^^" 前面的数字
                String numberStr = indexStr.substring(0, index);

                abnormalIndex = Integer.parseInt(numberStr);
            }else{
                throw new CustomHisAbnormalIndexException(observationType + "未查询到csm:hasAbnormalObsRecentIndex的属性值！");
            }
        }

        return abnormalIndex;
    }
}
