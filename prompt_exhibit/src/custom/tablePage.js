import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Paper,
  Typography,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
} from "@mui/material";
// 使用 Material-UI 的全局样式
import { GlobalStyles } from "@mui/system";

function TablePage() {
    // 分页设置
    const [page, setPage] = useState(0);
    const [rowsPerPage, setRowsPerPage] = useState(10);


    const [tableData, setTableData] = useState([]);
    const [dialogOpen, setDialogOpen] = useState(false);
    const [dialogContent, setDialogContent] = useState("");
    const [filterText, setFilterText] = useState("");
  
    useEffect(() => {
      fetchData();
    }, []);
  
    const fetchData = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:3001/data");
        setTableData(response.data);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };
  
    const handleDialogOpen = (content) => {
      setDialogContent(content);
      setDialogOpen(true);
    };
  
    const handleDialogClose = () => {
      setDialogOpen(false);
    };
  
    return (
      <div>
        <GlobalStyles styles={{
        ".MuiTableCell-root": {
          borderBottom: "1px solid rgba(224, 224, 224, 1)",
          borderRight: "1px solid rgba(224, 224, 224, 1)",
        },
      }}/>
        <Box sx={{width:"100%", overflowX:"auto", border:"1px solid rgba(224, 224, 224, 1)", padding:"16px",}}>
        <TablePagination 
        component="div"
        count={tableData.length}
        page={page}
        onPageChange={(event, newPage) => setPage(newPage)}
        rowsPerPage={rowsPerPage}
        onRowsPerPageChange={(event) =>
          setRowsPerPage(parseInt(event.target.value, 10))
        }
        labelRowsPerPage="每页行数:"
        />
        <TableContainer component={Paper}>
          <input
            type="text"
            placeholder="search"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
          />
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>序号</TableCell>
                <TableCell>请求 ID</TableCell>
                <TableCell>推理请求字符串</TableCell>
                <TableCell>第一个模型的推理结果</TableCell>
                <TableCell>第二个模型的推理结果</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {tableData.filter(
                (row) =>
                  row.prompt_id.includes(filterText) || row.prompt.includes(filterText)
              ).slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((row, index) => (
                <TableRow key={row.prompt_id}>
                  <TableCell>{page * rowsPerPage + index + 1}</TableCell>
                  <TableCell>{row.prompt_id}</TableCell>
                  <TableCell>
                    <Typography
                      onClick={() => handleDialogOpen(row.prompt)}
                      style={{ cursor: "pointer" }}
                    >
                      {row.prompt.substring(0, 100)}...
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography
                      onClick={() => handleDialogOpen(row.first_result)}
                      style={{ cursor: "pointer" }}
                    >
                      {row.first_result.substring(0, 100)}...
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography
                      onClick={() => handleDialogOpen(row.second_result)}
                      style={{ cursor: "pointer" }}
                    >
                      {row.second_result.substring(0, 100)}...
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        </Box>
        <Dialog open={dialogOpen} onClose={handleDialogClose}>
          <DialogTitle>详情</DialogTitle>
          <DialogContent>
            <Typography>{dialogContent}</Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleDialogClose}>关闭</Button>
          </DialogActions>
        </Dialog>
      </div>
    );
}

export default TablePage;