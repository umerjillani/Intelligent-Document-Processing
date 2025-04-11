// import { useState, useEffect } from "react"
// import {
//   Upload,
//   message,
//   Table,
//   Card,
//   Button,
//   Modal,
//   Form,
//   Select,
//   Space,
//   Spin,
//   Typography,
//   Descriptions,
//   List,
//   Tabs,
// } from "antd"
// import { InboxOutlined, FilterOutlined, CheckCircleOutlined, ExclamationCircleOutlined, CodeOutlined } from "@ant-design/icons"
// import axios from "axios"
// import ReactJson from "react-json-view"

// const { Dragger } = Upload
// const { Option } = Select
// const { Title, Text } = Typography
// const { TabPane } = Tabs

// const DocumentUpload = () => {
//   const [documents, setDocuments] = useState([])
//   const [fileList, setFileList] = useState([])
//   const [isFilterModalVisible, setIsFilterModalVisible] = useState(false)
//   const [isLoading, setIsLoading] = useState(false)
//   const [responseModalVisible, setResponseModalVisible] = useState(false)
//   const [apiResponse, setApiResponse] = useState(null)
//   const [jsonViewerVisible, setJsonViewerVisible] = useState(false)
//   const [selectedDocument, setSelectedDocument] = useState(null)

//   useEffect(() => {
//     const fetchDocuments = async () => {
//       try {
//         const response = await axios.get("http://127.0.0.1:3000/api/get_docs")
//         const rawData = response.data
//         console.log(response)
//         // Transform the nested arrays into objects
//         const formattedDocs = rawData.map((row) => ({
//           id: row[0],
//           doc_name: row[1],
//           batch_name: row[2],
//           type_name: row[3],
//           status: row[4],
//           created_at: row[5],
//           json_content: row[6] ? JSON.parse(row[6]) : null
//         }))

//         setDocuments(formattedDocs)
//         console.log(formattedDocs)
//       } catch (error) {
//         console.error("Error fetching documents:", error)
//         message.error("Failed to fetch documents from backend")
//       }
//     }

//     fetchDocuments()
//   }, [])

//   const handleUpload = (info) => {
//     // Store the complete file objects instead of just the paths
//     const newFileList = info.fileList
//     setFileList(newFileList)
//     console.log("Updated file list:", newFileList)
//   }

//   const handleSubmit = async () => {
//     try {
//       // Set loading state to true
//       setIsLoading(true)

//       // Extract files or file paths as needed for your backend
//       const files = fileList.map((file) => file.originFileObj)

//       console.log("Sending files to backend:", files)

//       // Create FormData for file upload
//       const formData = new FormData()
//       files.forEach((file) => {
//         formData.append("files", file)
//       })

//       const response = await axios.post("http://127.0.0.1:3000/api/process_docs", formData, {
//         headers: {
//           "Content-Type": "multipart/form-data",
//         },
//       })

//       console.log(response.data)

//       // Ensure the response data has the expected structure with proper defaults
//       const processedResponse = {
//         status_code: response.data.status_code || 500,
//         message: response.data.message || "No response message",
//         total_check_amount: response.data.total_check_amount || "0.00",
//         successful_docs: response.data.successful_docs,
//         exception_docs: response.data.exception_docs,
//         exceptions: Array.isArray(response.data.exceptions) ? response.data.exceptions : [],
//       }

//       // Store the processed API response
//       setApiResponse(processedResponse)

//       // Show the response modal
//       setResponseModalVisible(true)

//       // Clear the file list after successful upload
//       setFileList([])

//       // Refresh document list after processing
//       const refreshResponse = await axios.get("http://127.0.0.1:3000/api/get_docs")
//       const rawData = refreshResponse.data
//       const formattedDocs = rawData.map((row) => ({
//         id: row[0],
//         doc_name: row[1],
//         batch_name: row[2],
//         type_name: row[3],
//         status: row[4],
//         created_at: row[5],
//         json_content: row[6] ? JSON.parse(row[6]) : null
//       }))
//       setDocuments(formattedDocs)

//       message.success("Files processed successfully")
//     } catch (error) {
//       console.error("Upload error:", error)
//       message.error("Failed to send files to backend")
//     } finally {
//       // Set loading state to false regardless of success or failure
//       setIsLoading(false)
//     }
//   }

//   const viewDocumentJson = (record) => {
//     setSelectedDocument(record)
//     setJsonViewerVisible(true)
//   }

//   const columns = [
//     {
//       title: "Document Name",
//       dataIndex: "doc_name",
//       key: "doc_name",
//     },
//     {
//       title: "Batch",
//       dataIndex: "batch_name",
//       key: "batch_name",
//     },
//     {
//       title: "Type",
//       dataIndex: "type_name",
//       key: "type_name",
//       filters: [
//         { text: "Claim", value: "Claim" },
//         { text: "Agency", value: "Agency" },
//         { text: "Check", value: "Check" },
//       ],
//       onFilter: (value, record) => record.type_name.includes(value),
//     },
//     {
//       title: "Status",
//       dataIndex: "status",
//       key: "status",
//       render: (status) => {
//         const color = {
//           processed: "green",
//           exception: "red",
//           pending: "orange",
//           uploaded: "blue"
//         }[status.toLowerCase()] || "gray"
//         return <span style={{ color }}>{status}</span>
//       },
//     },
//     {
//       title: "Created At",
//       dataIndex: "created_at",
//       key: "created_at",
//     },
//     {
//       title: "Actions",
//       key: "actions",
//       render: (_, record) => (
//         <Space>
//           <Button 
//             size="small" 
//             onClick={() => viewDocumentJson(record)} 
//             disabled={!record.json_content}
//             icon={<CodeOutlined />}
//           >
//             View
//           </Button>
//           <Button size="small" danger>
//             Delete
//           </Button>
//         </Space>
//       ),
//     },
//   ]

//   // Combine mock documents with uploaded files
//   const tableData = [
//     ...documents.map((doc) => ({
//       key: doc.id,
//       ...doc,
//     })),
//   ]

//   const handleFilterSubmit = (values) => {
//     console.log("Filter values:", values)
//     setIsFilterModalVisible(false)
//   }

//   const closeResponseModal = () => {
//     setResponseModalVisible(false)
//     setApiResponse(null)
//   }

//   const closeJsonViewer = () => {
//     setJsonViewerVisible(false)
//     setSelectedDocument(null)
//   }

//   // Function to extract and display key information from JSON
//   const renderKeyInfo = (json) => {
//     if (!json || !json.Important_Info) return <Text>No important information available</Text>
    
//     const info = json.Important_Info
//     const entries = Object.entries(info)
    
//     return (
//       <List
//         size="small"
//         bordered
//         dataSource={entries}
//         renderItem={([key, value]) => (
//           <List.Item>
//             <Text strong>{key}:</Text> {value}
//           </List.Item>
//         )}
//       />
//     )
//   }

//   return (
//     <Card title="Document Upload">
//       <Spin spinning={isLoading} tip="Processing documents...">
//         <Dragger beforeUpload={() => false} onChange={handleUpload} multiple fileList={fileList} disabled={isLoading}>
//           <p className="ant-upload-drag-icon">
//             <InboxOutlined />
//           </p>
//           <p className="ant-upload-text">Click or drag files to upload</p>
//         </Dragger>
//         <Button
//           onClick={handleSubmit}
//           type="primary"
//           style={{ marginTop: 16 }}
//           disabled={fileList.length === 0 || isLoading}
//           loading={isLoading}
//         >
//           Process
//         </Button>

//         <div style={{ marginTop: 16, marginBottom: 16 }}>
//           <Button icon={<FilterOutlined />} onClick={() => setIsFilterModalVisible(true)} disabled={isLoading}>
//             Filters
//           </Button>
//         </div>

//         <Table columns={columns} dataSource={tableData} pagination={{ pageSize: 5 }} loading={isLoading} />
//       </Spin>

//       <Modal
//         title="Filter Documents"
//         visible={isFilterModalVisible}
//         onCancel={() => setIsFilterModalVisible(false)}
//         footer={null}
//       >
//         <Form onFinish={handleFilterSubmit}>
//           <Form.Item name="documentType" label="Document Type">
//             <Select placeholder="Select document type">
//               <Option value="Claim">Claim</Option>
//               <Option value="Agency">Agency</Option>
//               <Option value="Mortgage">Mortgage</Option>
//             </Select>
//           </Form.Item>
//           <Form.Item name="status" label="Status">
//             <Select placeholder="Select status">
//               <Option value="Uploaded">Uploaded</Option>
//               <Option value="Processing">Processing</Option>
//               <Option value="Completed">Completed</Option>
//               <Option value="Failed">Failed</Option>
//             </Select>
//           </Form.Item>
//           <Form.Item>
//             <Button type="primary" htmlType="submit">
//               Apply Filters
//             </Button>
//           </Form.Item>
//         </Form>
//       </Modal>

//       {/* Response Modal */}
//       <Modal
//         title={
//           <div style={{ display: "flex", alignItems: "center" }}>
//             {apiResponse?.status_code === 200 ? (
//               <CheckCircleOutlined style={{ color: "green", marginRight: 8 }} />
//             ) : (
//               <ExclamationCircleOutlined style={{ color: "red", marginRight: 8 }} />
//             )}
//             <span>Document Processing Result</span>
//           </div>
//         }
//         visible={responseModalVisible}
//         onCancel={closeResponseModal}
//         footer={[
//           <Button key="close" type="primary" onClick={closeResponseModal}>
//             Close
//           </Button>,
//         ]}
//         width={700}
//       >
//         {apiResponse && (
//           <div>
//             <Descriptions bordered column={1}>
//               <Descriptions.Item label="Status">
//                 <Text type={apiResponse.status_code === 200 ? "success" : "danger"}>{apiResponse.message}</Text>
//               </Descriptions.Item>
//               <Descriptions.Item label="Total Amount">${apiResponse.total_check_amount}</Descriptions.Item>
//             </Descriptions>

//             <div style={{ marginTop: 16 }}>
//              <Text type="success">Successful: {apiResponse.successful_docs}</Text>
//             </div>

//             <div style={{ marginTop: 16 }}>
//               <Text type="danger">Exceptions: {apiResponse.exceptions.length}</Text>
//             </div>

//             {apiResponse?.exceptions && Array.isArray(apiResponse.exceptions) && apiResponse.exceptions.length > 0 && (
//               <div style={{ marginTop: 16 }}>
//                 <Title level={5}>Exceptions</Title>
//                 <List
//                   size="small"
//                   bordered
//                   dataSource={apiResponse.exceptions}
//                   renderItem={(item) => (
//                     <List.Item>
//                       <Text type="danger">
//                         <strong>{item.filename}</strong>: {item.reason}
//                       </Text>
//                     </List.Item>
//                   )}
//                 />
//               </div>
//             )}
//           </div>
//         )}
//       </Modal>

//       {/* JSON Viewer Modal */}
//       <Modal
//         title={
//           <div style={{ display: "flex", alignItems: "center" }}>
//             <CodeOutlined style={{ marginRight: 8 }} />
//             <span>Document Details: {selectedDocument?.doc_name}</span>
//           </div>
//         }
//         visible={jsonViewerVisible}
//         onCancel={closeJsonViewer}
//         footer={[
//           <Button key="close" type="primary" onClick={closeJsonViewer}>
//             Close
//           </Button>,
//         ]}
//         width={800}
//       >
//         {selectedDocument && (
//           <div>
//             <Descriptions bordered column={1} style={{ marginBottom: 16 }}>
//               <Descriptions.Item label="Document Name">{selectedDocument.doc_name}</Descriptions.Item>
//               <Descriptions.Item label="Type">{selectedDocument.type_name}</Descriptions.Item>
//               <Descriptions.Item label="Status">
//                 <Text
//                   type={
//                     selectedDocument.status.toLowerCase() === "processed"
//                       ? "success"
//                       : selectedDocument.status.toLowerCase() === "exception"
//                       ? "danger"
//                       : "warning"
//                   }
//                 >
//                   {selectedDocument.status}
//                 </Text>
//               </Descriptions.Item>
//               <Descriptions.Item label="Batch">{selectedDocument.batch_name}</Descriptions.Item>
//               <Descriptions.Item label="Created At">{selectedDocument.created_at}</Descriptions.Item>
//             </Descriptions>

//             <Tabs defaultActiveKey="keyInfo">
//               <TabPane tab="Key Information" key="keyInfo">
//                 {selectedDocument.json_content ? (
//                   renderKeyInfo(selectedDocument.json_content)
//                 ) : (
//                   <Text type="secondary">No JSON data available for this document</Text>
//                 )}
//               </TabPane>
//               <TabPane tab="Raw JSON" key="rawJson">
//                 {selectedDocument.json_content ? (
//                   <div style={{ backgroundColor: "#f5f5f5", padding: 16, borderRadius: 4, maxHeight: "400px", overflow: "auto" }}>
//                     <ReactJson
//                       src={selectedDocument.json_content}
//                       theme="rjv-default"
//                       displayDataTypes={false}
//                       name={false}
//                       collapsed={1}
//                       enableClipboard={true}
//                     />
//                   </div>
//                 ) : (
//                   <Text type="secondary">No JSON data available for this document</Text>
//                 )}
//               </TabPane>
//             </Tabs>
//           </div>
//         )}
//       </Modal>
//     </Card>
//   )
// }

// export default DocumentUpload

import { useState, useEffect } from "react"
import {
  Upload,
  message,
  Table,
  Card,
  Button,
  Modal,
  Form,
  Select,
  Space,
  Spin,
  Typography,
  Descriptions,
  List,
  Tabs,
} from "antd"
import { InboxOutlined, FilterOutlined, CheckCircleOutlined, ExclamationCircleOutlined, CodeOutlined, SaveOutlined } from "@ant-design/icons"
import axios from "axios"
import ReactJson from "react-json-view"

const { Dragger } = Upload
const { Option } = Select
const { Title, Text } = Typography
const { TabPane } = Tabs

const DocumentUpload = () => {
  const [documents, setDocuments] = useState([])
  const [fileList, setFileList] = useState([])
  const [isFilterModalVisible, setIsFilterModalVisible] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [responseModalVisible, setResponseModalVisible] = useState(false)
  const [apiResponse, setApiResponse] = useState(null)
  const [jsonViewerVisible, setJsonViewerVisible] = useState(false)
  const [selectedDocument, setSelectedDocument] = useState(null)
  const [editedJson, setEditedJson] = useState(null)
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:3000/api/get_docs")
        const rawData = response.data
        console.log(response)
        // Transform the nested arrays into objects
        const formattedDocs = rawData.map((row) => ({
          id: row[0],
          doc_name: row[1],
          batch_name: row[2],
          type_name: row[3],
          status: row[4],
          created_at: row[5],
          json_content: row[6] ? JSON.parse(row[6]) : null
        }))

        setDocuments(formattedDocs)
        console.log(formattedDocs)
      } catch (error) {
        console.error("Error fetching documents:", error)
        message.error("Failed to fetch documents from backend")
      }
    }

    fetchDocuments()
  }, [])

  const handleUpload = (info) => {
    // Store the complete file objects instead of just the paths
    const newFileList = info.fileList
    setFileList(newFileList)
    console.log("Updated file list:", newFileList)
  }

  const handleSubmit = async () => {
    try {
      // Set loading state to true
      setIsLoading(true)

      // Extract files or file paths as needed for your backend
      const files = fileList.map((file) => file.originFileObj)

      console.log("Sending files to backend:", files)

      // Create FormData for file upload
      const formData = new FormData()
      files.forEach((file) => {
        formData.append("files", file)
      })

      const response = await axios.post("http://127.0.0.1:3000/api/process_docs", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })

      console.log(response.data)

      // Ensure the response data has the expected structure with proper defaults
      const processedResponse = {
        status_code: response.data.status_code || 500,
        message: response.data.message || "No response message",
        total_check_amount: response.data.total_check_amount || "0.00",
        successful_docs: response.data.successful_docs,
        exception_docs: response.data.exception_docs,
        exceptions: Array.isArray(response.data.exceptions) ? response.data.exceptions : [],
      }

      // Store the processed API response
      setApiResponse(processedResponse)

      // Show the response modal
      setResponseModalVisible(true)

      // Clear the file list after successful upload
      setFileList([])

      // Refresh document list after processing
      const refreshResponse = await axios.get("http://127.0.0.1:3000/api/get_docs")
      const rawData = refreshResponse.data
      const formattedDocs = rawData.map((row) => ({
        id: row[0],
        doc_name: row[1],
        batch_name: row[2],
        type_name: row[3],
        status: row[4],
        created_at: row[5],
        json_content: row[6] ? JSON.parse(row[6]) : null
      }))
      setDocuments(formattedDocs)

      message.success("Files processed successfully")
    } catch (error) {
      console.error("Upload error:", error)
      message.error("Failed to send files to backend")
    } finally {
      // Set loading state to false regardless of success or failure
      setIsLoading(false)
    }
  }

  const viewDocumentJson = (record) => {
    setSelectedDocument(record)
    setEditedJson(record.json_content)
    setJsonViewerVisible(true)
  }

  const handleJsonEdit = (edit) => {
    setEditedJson(edit.updated_src)
  }

  const saveJsonChanges = async () => {
    console.log('Idhr')
    if (!selectedDocument || !editedJson) return
    
    setIsSaving(true)
    try {
      // Make API call to save the updated JSON
      const response = await axios.post("http://127.0.0.1:3000/api/update_json", {
        document_id: selectedDocument.id,
        json_content: editedJson
      })
      
      // Update the documents array with the new JSON
      const updatedDocuments = documents.map(doc => {
        if (doc.id === selectedDocument.id) {
          return { ...doc, json_content: editedJson }
        }
        return doc
      })
      
      setDocuments(updatedDocuments)
      
      // Update selected document
      setSelectedDocument({ ...selectedDocument, json_content: editedJson })
      
      message.success("JSON updated successfully")
    } catch (error) {
      console.error("Error updating JSON:", error)
      message.error("Failed to update JSON data")
    } finally {
      setIsSaving(false)
    }
  }

  const columns = [
    {
      title: "Document Name",
      dataIndex: "doc_name",
      key: "doc_name",
    },
    {
      title: "Batch",
      dataIndex: "batch_name",
      key: "batch_name",
    },
    {
      title: "Type",
      dataIndex: "type_name",
      key: "type_name",
      filters: [
        { text: "Claim", value: "Claim" },
        { text: "Agency", value: "Agency" },
        { text: "Check", value: "Check" },
      ],
      onFilter: (value, record) => record.type_name.includes(value),
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      render: (status) => {
        const color = {
          processed: "green",
          exception: "red",
          pending: "orange",
          uploaded: "blue"
        }[status.toLowerCase()] || "gray"
        return <span style={{ color }}>{status}</span>
      },
    },
    {
      title: "Created At",
      dataIndex: "created_at",
      key: "created_at",
    },
    {
      title: "Actions",
      key: "actions",
      render: (_, record) => (
        <Space>
          <Button 
            size="small" 
            onClick={() => viewDocumentJson(record)} 
            disabled={!record.json_content}
            icon={<CodeOutlined />}
          >
            View
          </Button>
          <Button size="small" danger>
            Delete
          </Button>
        </Space>
      ),
    },
  ]

  // Combine mock documents with uploaded files
  const tableData = [
    ...documents.map((doc) => ({
      key: doc.id,
      ...doc,
    })),
  ]

  const handleFilterSubmit = (values) => {
    console.log("Filter values:", values)
    setIsFilterModalVisible(false)
  }

  const closeResponseModal = () => {
    setResponseModalVisible(false)
    setApiResponse(null)
  }

  const closeJsonViewer = () => {
    setJsonViewerVisible(false)
    setSelectedDocument(null)
    setEditedJson(null)
  }

  // Function to extract and display key information from JSON
  const renderKeyInfo = (json) => {
    if (!json || !json.Important_Info) return <Text>No important information available</Text>
    
    const info = json.Important_Info
    const entries = Object.entries(info)
    
    return (
      <List
        size="small"
        bordered
        dataSource={entries}
        renderItem={([key, value]) => (
          <List.Item>
            <Text strong>{key}:</Text> {value}
          </List.Item>
        )}
      />
    )
  }

  return (
    <Card title="Document Upload">
      <Spin spinning={isLoading} tip="Processing documents...">
        <Dragger beforeUpload={() => false} onChange={handleUpload} multiple fileList={fileList} disabled={isLoading}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">Click or drag files to upload</p>
        </Dragger>
        <Button
          onClick={handleSubmit}
          type="primary"
          style={{ marginTop: 16 }}
          disabled={fileList.length === 0 || isLoading}
          loading={isLoading}
        >
          Process
        </Button>

        <div style={{ marginTop: 16, marginBottom: 16 }}>
          <Button icon={<FilterOutlined />} onClick={() => setIsFilterModalVisible(true)} disabled={isLoading}>
            Filters
          </Button>
        </div>

        <Table columns={columns} dataSource={tableData} pagination={{ pageSize: 5 }} loading={isLoading} />
      </Spin>

      <Modal
        title="Filter Documents"
        visible={isFilterModalVisible}
        onCancel={() => setIsFilterModalVisible(false)}
        footer={null}
      >
        <Form onFinish={handleFilterSubmit}>
          <Form.Item name="documentType" label="Document Type">
            <Select placeholder="Select document type">
              <Option value="Claim">Claim</Option>
              <Option value="Agency">Agency</Option>
              <Option value="Mortgage">Mortgage</Option>
            </Select>
          </Form.Item>
          <Form.Item name="status" label="Status">
            <Select placeholder="Select status">
              <Option value="Uploaded">Uploaded</Option>
              <Option value="Processing">Processing</Option>
              <Option value="Completed">Completed</Option>
              <Option value="Failed">Failed</Option>
            </Select>
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit">
              Apply Filters
            </Button>
          </Form.Item>
        </Form>
      </Modal>

      {/* Response Modal */}
      <Modal
        title={
          <div style={{ display: "flex", alignItems: "center" }}>
            {apiResponse?.status_code === 200 ? (
              <CheckCircleOutlined style={{ color: "green", marginRight: 8 }} />
            ) : (
              <ExclamationCircleOutlined style={{ color: "red", marginRight: 8 }} />
            )}
            <span>Document Processing Result</span>
          </div>
        }
        visible={responseModalVisible}
        onCancel={closeResponseModal}
        footer={[
          <Button key="close" type="primary" onClick={closeResponseModal}>
            Close
          </Button>,
        ]}
        width={700}
      >
        {apiResponse && (
          <div>
            <Descriptions bordered column={1}>
              <Descriptions.Item label="Status">
                <Text type={apiResponse.status_code === 200 ? "success" : "danger"}>{apiResponse.message}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="Total Amount">${apiResponse.total_check_amount}</Descriptions.Item>
            </Descriptions>

            <div style={{ marginTop: 16 }}>
             <Text type="success">Successful: {apiResponse.successful_docs}</Text>
            </div>

            <div style={{ marginTop: 16 }}>
              <Text type="danger">Exceptions: {apiResponse.exceptions.length}</Text>
            </div>

            {apiResponse?.exceptions && Array.isArray(apiResponse.exceptions) && apiResponse.exceptions.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <Title level={5}>Exceptions</Title>
                <List
                  size="small"
                  bordered
                  dataSource={apiResponse.exceptions}
                  renderItem={(item) => (
                    <List.Item>
                      <Text type="danger">
                        <strong>{item.filename}</strong>: {item.reason}
                      </Text>
                    </List.Item>
                  )}
                />
              </div>
            )}
          </div>
        )}
      </Modal>

      {/* JSON Viewer/Editor Modal */}
      <Modal
        title={
          <div style={{ display: "flex", alignItems: "center" }}>
            <CodeOutlined style={{ marginRight: 8 }} />
            <span>Document Details: {selectedDocument?.doc_name}</span>
          </div>
        }
        visible={jsonViewerVisible}
        onCancel={closeJsonViewer}
        footer={[
          <Button 
            key="save" 
            type="primary" 
            icon={<SaveOutlined />} 
            onClick={saveJsonChanges}
            loading={isSaving}
            disabled={!editedJson}
            style={{ marginRight: 8 }}
          >
            Save Changes
          </Button>,
          <Button key="close" onClick={closeJsonViewer}>
            Close
          </Button>,
        ]}
        width={800}
      >
        {selectedDocument && (
          <div>
            <Descriptions bordered column={1} style={{ marginBottom: 16 }}>
              <Descriptions.Item label="Document Name">{selectedDocument.doc_name}</Descriptions.Item>
              <Descriptions.Item label="Type">{selectedDocument.type_name}</Descriptions.Item>
              <Descriptions.Item label="Status">
                <Text
                  type={
                    selectedDocument.status.toLowerCase() === "processed"
                      ? "success"
                      : selectedDocument.status.toLowerCase() === "exception"
                      ? "danger"
                      : "warning"
                  }
                >
                  {selectedDocument.status}
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="Batch">{selectedDocument.batch_name}</Descriptions.Item>
              <Descriptions.Item label="Created At">{selectedDocument.created_at}</Descriptions.Item>
            </Descriptions>

            <Tabs defaultActiveKey="keyInfo">
              <TabPane tab="Key Information" key="keyInfo">
                {editedJson ? (
                  renderKeyInfo(editedJson)
                ) : (
                  <Text type="secondary">No JSON data available for this document</Text>
                )}
              </TabPane>
              <TabPane tab="Edit JSON" key="editJson">
                {editedJson ? (
                  <div style={{ backgroundColor: "#f5f5f5", padding: 16, borderRadius: 4, maxHeight: "400px", overflow: "auto" }}>
                    <ReactJson
                      src={editedJson}
                      theme="rjv-default"
                      displayDataTypes={false}
                      name={false}
                      collapsed={1}
                      enableClipboard={true}
                      onEdit={handleJsonEdit}
                      onAdd={handleJsonEdit}
                      onDelete={handleJsonEdit}
                    />
                  </div>
                ) : (
                  <Text type="secondary">No JSON data available for this document</Text>
                )}
              </TabPane>
            </Tabs>
          </div>
        )}
      </Modal>
    </Card>
  )
}

export default DocumentUpload