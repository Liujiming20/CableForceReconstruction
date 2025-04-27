package com.sjtu.KGStudy.abnormalDataRec.utils;

import com.sjtu.KGStudy.utils.PrefixesDel;
import org.apache.jena.query.*;

import org.apache.jena.rdf.model.*;
import org.apache.jena.vocabulary.RDF;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class ObsStatusChecking {
    public static Model getQueryResult(Model delModel){
        // 创建一个新的RDF模型用于存储查询结果
        Model queryResultModel = ModelFactory.createDefaultModel();

        StringBuffer stringBufferQuery = PrefixesDel.getPrefixes(delModel);
        String queryBody = "SELECT ?cableTensionObs (COUNT(?envTempObs) AS ?resultCount) \n" +
                "WHERE{{?cableTensionObs a csm:CableTensionObservation;" +
                "      csm:hasCurrentObservationStatus false.}" +
                "OPTIONAL {?envTempObs a csm:EnvironmentalTemperatureObservation;" +
                "      csm:hasAbnormalObsRecentIndex ?abnormalIndex." +
                "      FILTER(?abnormalIndex > '-1'^^xsd:integer)}}"+
                "GROUP BY ?cableTensionObs";

        stringBufferQuery.append(queryBody);
        Query querySPARQL = QueryFactory.create(stringBufferQuery.toString());

        try (QueryExecution qexec = QueryExecutionFactory.create(querySPARQL, delModel)) {
            ResultSet results = qexec.execSelect();

            // 复制原始模型中的命名空间到查询结果模型中
            delModel.getNsPrefixMap().forEach((prefix, uri) -> queryResultModel.setNsPrefix(prefix, uri));

            // 定义命名空间
            String csm = "http://www.semanticweb.org/16648/ontologies/2023/2/CableStructureMonitor#";

            // 为创建rdf:type三元组生成类
            Resource cableTensionClass = queryResultModel.createResource(csm + "CableTensionObservation");

            // 创建数据对象（data property）
            Property predicate = queryResultModel.createProperty(csm, "correspondAbnormalHisObsNum");
            // 将查询结果添加到模型中
            while (results.hasNext()) {
                QuerySolution solution = results.nextSolution();
                Resource subject = solution.getResource("cableTensionObs");

                RDFNode object = solution.get("resultCount");

                // 添加查询结果到查询结果模型
                queryResultModel.add(subject, predicate, object);
                queryResultModel.add(subject, RDF.type, cableTensionClass);
                // 保存模型到本地文件
                try (FileOutputStream out = new FileOutputStream("G:/KG_study/reason_main/src/main/resources/abnormalDataRecServiceData/recModeSourceData.ttl")) {
                    queryResultModel.write(out, "Turtle");  // 输出为Turtle格式
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return queryResultModel;
    }

    // 根据csm:hasCurrentObservationStatus获悉观测异常的索张力传感器
    public static void queryTest(Model delModel) {
        StringBuffer stringBufferQuery = PrefixesDel.getPrefixes(delModel);
        /*String queryBody = "SELECT ?cableTensionObs ?envTempObs \n" +
                "WHERE{{?cableTensionObs a csm:CableTensionObservation;\n " +
                "      csm:hasCurrentObservationStatus false.}" +
                "UNION {?envTempObs a csm:EnvironmentalTemperatureObservation;" +
                "      csm:hasAbnormalObsRecentIndex ?abnormalIndex." +
                "      FILTER(?abnormalIndex > \'-1\'^^xsd:integer)}}";*/

        String queryBody = "SELECT ?cableTensionObs (COUNT(?envTempObs) AS ?resultCount) \n" +
                "WHERE{{?cableTensionObs a csm:CableTensionObservation;" +
                "      csm:hasCurrentObservationStatus false.}" +
                "OPTIONAL {?envTempObs a csm:EnvironmentalTemperatureObservation;" +
                "      csm:hasAbnormalObsRecentIndex ?abnormalIndex." +
                "      FILTER(?abnormalIndex > '-1'^^xsd:integer)}}"+
                "GROUP BY ?cableTensionObs";
        /*String queryBody = "SELECT ?this \n" +
                "    WHERE {         \n" +
                "        {SELECT ?this (COUNT(?envTempObs) AS ?resultCount)                    \n" +
                "            WHERE {\n" +
                "                {?this a csm:CableTensionObservation ;\n" +
                "                        csm:hasCurrentObservationStatus false .}\n" +
                "               OPTIONAL {?envTempObs a csm:EnvironmentalTemperatureObservation ;\n" +
                "                                     csm:hasAbnormalObsRecentIndex ?abnormalIndex .\n" +
                "                        FILTER (?abnormalIndex > '-1'^^xsd:integer)}}                    \n" +
                "                      GROUP BY ?this} \n" +
                "        FILTER (?resultCount = 1  || \n" +
                "                ?resultCount > 3 )}";*/

        stringBufferQuery.append(queryBody);
        Query querySPARQL = QueryFactory.create(stringBufferQuery.toString());

        // 打印完整查询语句
        System.out.println("SPARQL Query:");
        System.out.println(querySPARQL);

        try (QueryExecution qexec = QueryExecutionFactory.create(querySPARQL, delModel)) {
            ResultSet results = qexec.execSelect();

            // 打印所有查询结果
            System.out.println("Query Results:");
            ResultSetFormatter.out(System.out, results, querySPARQL);
        }
    }
}
