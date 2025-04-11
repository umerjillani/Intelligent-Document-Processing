import React, { useState } from 'react';
import { Layout, Menu } from 'antd';
import { 
  DashboardOutlined, 
  FileAddOutlined, 
  FolderOutlined, 
  FileDoneOutlined, 
  ExceptionOutlined 
} from '@ant-design/icons';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';

// Import all components
import Dashboard from './components/Dashboard';
import DocumentUpload from './components/DocumentUpload';
import BatchManagement from './components/BatchManagement';
import ProcessedDocuments from './components/ProcessedDocuments';
import Exceptions from './components/Exceptions';

const { Sider, Content } = Layout;

const App = () => {
  const [collapsed, setCollapsed] = useState(false);

  const menuItems = [
    {
      key: '1',
      icon: <DashboardOutlined />,
      label: <Link to="/">Dashboard</Link>
    },
    {
      key: '2',
      icon: <FileAddOutlined />,
      label: <Link to="/document-upload">Document Upload</Link>
    },
    {
      key: '3',
      icon: <FolderOutlined />,
      label: <Link to="/batch-management">Batch Management</Link>
    },
    {
      key: '4',
      icon: <FileDoneOutlined />,
      label: <Link to="/processed-documents">Processed Documents</Link>
    },
    {
      key: '5',
      icon: <ExceptionOutlined />,
      label: <Link to="/exceptions">Exceptions</Link>
    }
  ];

  return (
    <Router>
      <Layout style={{ minHeight: '100vh' }}>
        <Sider 
          collapsible 
          collapsed={collapsed} 
          onCollapse={(value) => setCollapsed(value)}
        >
          <div style={{ 
            height: 32, 
            margin: 16, 
            background: 'rgba(255, 255, 255, 0.2)' 
          }} />
          <Menu 
            theme="dark" 
            defaultSelectedKeys={['1']} 
            mode="inline" 
            items={menuItems}
          />
        </Sider>
        <Layout>
          <Content style={{ margin: '16px' }}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/document-upload" element={<DocumentUpload />} />
              <Route path="/batch-management" element={<BatchManagement />} />
              <Route path="/processed-documents" element={<ProcessedDocuments />} />
              <Route path="/exceptions" element={<Exceptions />} />
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </Router>
  );
};

export default App;