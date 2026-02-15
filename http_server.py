import asyncio
from http_server import start_http_server

# In your async main or run_forever method:
async def main():
    # Start HTTP dashboards (Railway's PORT)
    asyncio.create_task(start_http_server())
    
    # Your existing cognitive mesh on ZeroMQ (5555, 5556)
    mesh = CognitiveMesh(node_id="global_mind_01")
    await mesh.run_forever()

if __name__ == "__main__":
    asyncio.run(main())