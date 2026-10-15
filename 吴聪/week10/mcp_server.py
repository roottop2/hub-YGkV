from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

@mcp.tool
def add(a: float, b: float) -> float:
    """计算a和b的和"""
    return a + b

@mcp.tool
def sub(a: float, b: float) -> float:
    """计算a和b的差值"""
    return a - b

@mcp.tool
def multiply(a: float, b: float) -> float:
    """计算a和b的乘积"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)